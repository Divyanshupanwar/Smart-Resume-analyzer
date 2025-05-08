import streamlit as st
import nltk
import spacy
nltk.download('stopwords')
spacy.load('en_core_web_sm')

import pandas as pd
import base64, random
import time, datetime
from pyresparser import ResumeParser
from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
import io, random
from streamlit_tags import st_tags
from PIL import Image
import pymysql
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
import plotly.express as px
import numpy as np
import yt_dlp
import os

# DAA Implementations
class SortingAlgorithms:
    @staticmethod
    def merge_sort(arr, key=lambda x: x):
        """
        Merge sort implementation for sorting resumes
        Time Complexity: O(n log n)
        """
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = SortingAlgorithms.merge_sort(arr[:mid], key)
        right = SortingAlgorithms.merge_sort(arr[mid:], key)
        
        return SortingAlgorithms._merge(left, right, key)
    
    @staticmethod
    def _merge(left, right, key):
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if key(left[i]) >= key(right[j]):   # Using >= for descending order (higher scores first)
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    @staticmethod
    def quick_sort(arr, key=lambda x: x):
        """
        Quick sort implementation for sorting resumes
        Time Complexity: Average O(n log n), Worst O(n²)
        """
        if len(arr) <= 1:
            return arr
        
        pivot = arr[len(arr) // 2]
        pivot_val = key(pivot)
        
        left = [x for x in arr if key(x) > pivot_val]  # Greater values (for descending order)
        middle = [x for x in arr if key(x) == pivot_val]
        right = [x for x in arr if key(x) < pivot_val]  # Lesser values
        
        return SortingAlgorithms.quick_sort(left, key) + middle + SortingAlgorithms.quick_sort(right, key)
    
    @staticmethod
    def heap_sort(arr, key=lambda x: x):
        """
        Heap sort implementation for sorting resumes
        Time Complexity: O(n log n)
        """
        def heapify(arr, n, i):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n and key(arr[largest]) < key(arr[left]):
                largest = left
                
            if right < n and key(arr[largest]) < key(arr[right]):
                largest = right
                
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(arr, n, largest)
        
        n = len(arr)
        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)
            
        # Extract elements one by one
        result = []
        for i in range(n - 1, -1, -1):
            arr[0], arr[i] = arr[i], arr[0]
            result.insert(0, arr[i])  # Insert at beginning for descending order
            heapify(arr, i, 0)
            
        return result

class KnapsackAlgorithm:
    @staticmethod
    def knapsack_dp(values, weights, capacity):
        """
        Dynamic Programming approach to 0/1 Knapsack Problem
        Used to select optimal resumes based on scores (values) and constraints (weights)
        Time Complexity: O(n*W) where n is number of items and W is capacity
        """
        n = len(values)
        # Initialize DP table
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        # Build table in bottom-up manner
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
                else:
                    dp[i][w] = dp[i-1][w]
        
        # Find selected items
        selected = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected.append(i-1)
                w -= weights[i-1]
        
        return dp[n][capacity], selected

class ResumeRanker:
    @staticmethod
    def calculate_resume_score(resume_data, resume_text, skills_match_score):
        """
        Calculate comprehensive score for a resume based on multiple factors
        """
        score = 0
        
        # Base score from resume parser
        if 'Objective' in resume_text:
            score += 20
        if 'Declaration' in resume_text:
            score += 20
        if 'Hobbies' in resume_text or 'Interests' in resume_text:
            score += 20
        if 'Achievements' in resume_text:
            score += 20
        if 'Projects' in resume_text:
            score += 20
            
        # Experience level score
        pages = resume_data['no_of_pages']
        if pages == 1:
            score += 10  # Fresher
        elif pages == 2:
            score += 20  # Intermediate
        else:
            score += 30  # Experienced
            
        # Skills match score (calculated externally)
        score += skills_match_score
        
        # Education score (if available)
        if 'education' in resume_data and resume_data['education']:
            score += 15
            
        # Experience score (if available)
        if 'experience' in resume_data and resume_data['experience']:
            # Add 5 points per year of experience, up to 25
            exp_years = min(5, len(resume_data['experience']))
            score += exp_years * 5
            
        return score

    @staticmethod
    def select_best_resumes(resumes, algorithm='merge_sort'):
        """
        Sort and select the best resumes using the specified algorithm
        """
        if algorithm == 'merge_sort':
            return SortingAlgorithms.merge_sort(resumes, key=lambda x: x['total_score'])
        elif algorithm == 'quick_sort':
            return SortingAlgorithms.quick_sort(resumes, key=lambda x: x['total_score'])
        elif algorithm == 'heap_sort':
            return SortingAlgorithms.heap_sort(resumes, key=lambda x: x['total_score'])
        else:
            raise ValueError(f"Unknown sorting algorithm: {algorithm}")
    
    @staticmethod
    def select_optimal_resumes(resumes, max_candidates=10, diversity_weight=0.3):
        """
        Use knapsack algorithm to select optimal set of resumes based on 
        score and diversity constraints
        """
        if not resumes:
            return []
            
        # Extract values (scores) and weights (diversity measures)
        values = [r['total_score'] for r in resumes]
        
        # Calculate diversity weights based on skills and fields
        # Higher weight means more diverse candidate (helps with team composition)
        weights = []
        skill_sets = [set(r['skills']) for r in resumes]
        fields = [r['predicted_field'] for r in resumes]
        
        for i, resume in enumerate(resumes):
            # Calculate diversity as a function of unique skills compared to others
            skill_diversity = sum(len(skill_sets[i] - skill_sets[j]) 
                                for j in range(len(resumes)) if i != j)
            
            # Field diversity (bonus for underrepresented fields)
            field_count = fields.count(resume['predicted_field'])
            field_diversity = max(1, 10 - field_count)  # Higher for rare fields
            
            # Combined diversity weight (normalize to range 1-10)
            diversity = 1 + (skill_diversity / 10) + field_diversity
            weights.append(int(diversity))
        
        # Use knapsack to select optimal combination
        capacity = max_candidates * 5  # Adjust capacity based on max candidates
        max_value, selected_indices = KnapsackAlgorithm.knapsack_dp(values, weights, capacity)
        
        # Return selected resumes
        return [resumes[i] for i in selected_indices]

def fetch_yt_video(link):
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(link, download=False)
        video_title = info_dict.get('title', None)
        
    return video_title

def get_table_download_link(df, filename, text):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                    caching=True,
                                    check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()
    return text

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def course_recommender(course_list):
    st.subheader("**Courses & Certificates🎓 Recommendations**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 4)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course

# Database connection (modified to use a more flexible connection approach)
def create_db_connection():
    try:
        connection = pymysql.connect(
            host='localhost', 
            user='root', 
            password='div230111017@',
            database='sra'
        )
        return connection
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return None

# Modified database functions
def setup_database():
    try:
        connection = pymysql.connect(host='localhost', user='root', password='div230111017@')
        cursor = connection.cursor()
        
        # Create database if not exists
        cursor.execute("CREATE DATABASE IF NOT EXISTS SRA;")
        connection.select_db("sra")
        
        # Create user_data table with additional 'selected' column
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            ID INT NOT NULL AUTO_INCREMENT,
            Name varchar(100) NOT NULL,
            Email_ID VARCHAR(50) NOT NULL,
            resume_score VARCHAR(8) NOT NULL,
            Timestamp VARCHAR(50) NOT NULL,
            Page_no VARCHAR(5) NOT NULL,
            Predicted_Field VARCHAR(25) NOT NULL,
            User_level VARCHAR(30) NOT NULL,
            Actual_skills VARCHAR(300) NOT NULL,
            Recommended_skills VARCHAR(300) NOT NULL,
            Recommended_courses VARCHAR(600) NOT NULL,
            total_score INT DEFAULT 0,
            selected BOOLEAN DEFAULT FALSE,
            PRIMARY KEY (ID)
        );
        """)
        
        connection.commit()
        return connection, cursor
    except Exception as e:
        st.error(f"Database Setup Error: {e}")
        return None, None

def insert_data(cursor, connection, name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, 
               skills, recommended_skills, courses, total_score=0, selected=False):
    try:
        insert_sql = """
        INSERT INTO user_data (
            Name, Email_ID, resume_score, Timestamp, Page_no, 
            Predicted_Field, User_level, Actual_skills, 
            Recommended_skills, Recommended_courses, total_score, selected
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        rec_values = (
            name, email, str(res_score), timestamp, str(no_of_pages), 
            reco_field, cand_level, skills, recommended_skills, 
            courses, total_score, selected
        )
        
        cursor.execute(insert_sql, rec_values)
        connection.commit()
        return True
    except Exception as e:
        st.error(f"Database Insert Error: {e}")
        return False

def get_all_resumes(cursor):
    try:
        cursor.execute("""
        SELECT ID, Name, Email_ID, resume_score, Predicted_Field, 
               User_level, Actual_skills, total_score, selected 
        FROM user_data
        """)
        
        columns = ['id', 'name', 'email', 'resume_score', 'predicted_field', 
                  'experience_level', 'skills', 'total_score', 'selected']
        
        results = cursor.fetchall()
        resumes = []
        
        for row in results:
            resume_dict = dict(zip(columns, row))
            # Convert skills string to list
            if isinstance(resume_dict['skills'], str):
                resume_dict['skills'] = resume_dict['skills'].strip('[]').replace("'", "").split(', ')
            resumes.append(resume_dict)
            
        return resumes
    except Exception as e:
        st.error(f"Database Query Error: {e}")
        return []

def update_selected_status(cursor, connection, id_list, selected=True):
    try:
        # First reset all to not selected
        if selected:
            cursor.execute("UPDATE user_data SET selected = FALSE")
        
        # Then set selected for the chosen ones
        if id_list:
            placeholders = ', '.join(['%s'] * len(id_list))
            cursor.execute(f"UPDATE user_data SET selected = {selected} WHERE ID IN ({placeholders})", id_list)
            
        connection.commit()
        return True
    except Exception as e:
        st.error(f"Database Update Error: {e}")
        return False

def run():
    st.title("Enhanced Smart Resume Analyser")
    st.sidebar.markdown("# Choose User")
    activities = ["Normal User", "Admin"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    
    img = Image.open('./Logo/new_logo.jpg') if 'new_logo.jpg' in [f for f in os.listdir('./Logo')] else None
    if img:
        img = img.resize((250, 250))
        st.image(img)

    # Setup database
    connection, cursor = setup_database()
    if not connection or not cursor:
        st.error("Failed to connect to database. Please check your configuration.")
        return

    if choice == 'Normal User':
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        if pdf_file is not None:
            save_image_path = './Uploaded_Resumes/' + pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)
            resume_data = ResumeParser(save_image_path).get_extracted_data()
            if resume_data:
                # Get the whole resume text
                resume_text = pdf_reader(save_image_path)

                st.header("**Resume Analysis**")
                st.success("Hello " + resume_data['name'])
                st.subheader("**Your Basic info**")
                try:
                    st.text('Name: ' + resume_data['name'])
                    st.text('Email: ' + resume_data['email'])
                    st.text('Contact: ' + resume_data['mobile_number'])
                    st.text('Resume pages: ' + str(resume_data['no_of_pages']))
                except:
                    pass
                
                # Determine candidate level
                cand_level = ''
                if resume_data['no_of_pages'] == 1:
                    cand_level = "Fresher"
                    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>You are looking Fresher.</h4>''',
                                unsafe_allow_html=True)
                elif resume_data['no_of_pages'] == 2:
                    cand_level = "Intermediate"
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',
                                unsafe_allow_html=True)
                elif resume_data['no_of_pages'] >= 3:
                    cand_level = "Experienced"
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',
                                unsafe_allow_html=True)

                st.subheader("**Skills Recommendation💡**")
                # Skills shows
                keywords = st_tags(label='### Skills that you have',
                                   text='See our skills recommendation',
                                   value=resume_data['skills'], key='1')

                # Skills categories
                ds_keyword = ['tensorflow', 'keras', 'pytorch', 'machine learning', 'deep Learning', 'flask',
                              'streamlit']
                web_keyword = ['react', 'django', 'node jS', 'react js', 'php', 'laravel', 'magento', 'wordpress',
                               'javascript', 'angular js', 'c#', 'flask']
                android_keyword = ['android', 'android development', 'flutter', 'kotlin', 'xml', 'kivy']
                ios_keyword = ['ios', 'ios development', 'swift', 'cocoa', 'cocoa touch', 'xcode']
                uiux_keyword = ['ux', 'adobe xd', 'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 'wireframes',
                                'storyframes', 'adobe photoshop', 'photoshop', 'editing', 'adobe illustrator',
                                'illustrator', 'adobe after effects', 'after effects', 'adobe premier pro',
                                'premier pro', 'adobe indesign', 'indesign', 'wireframe', 'solid', 'grasp',
                                'user research', 'user experience']

                recommended_skills = []
                reco_field = ''
                rec_course = ''
                skills_match_score = 0
                
                # Field recommendation based on skills
                for i in resume_data['skills']:
                    # Data science recommendation
                    if i.lower() in ds_keyword:
                        skills_match_score += 10
                        reco_field = 'Data Science'
                        st.success("** Our analysis says you are looking for Data Science Jobs.**")
                        recommended_skills = ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling',
                                              'Data Mining', 'Clustering & Classification', 'Data Analytics',
                                              'Quantitative Analysis', 'Web Scraping', 'ML Algorithms', 'Keras',
                                              'Pytorch', 'Probability', 'Scikit-learn', 'Tensorflow', "Flask",
                                              'Streamlit']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                       text='Recommended skills generated from System',
                                                       value=recommended_skills, key='2')
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(ds_course)
                        break

                    # Web development recommendation
                    elif i.lower() in web_keyword:
                        skills_match_score += 10
                        reco_field = 'Web Development'
                        st.success("** Our analysis says you are looking for Web Development Jobs **")
                        recommended_skills = ['React', 'Django', 'Node JS', 'React JS', 'php', 'laravel', 'Magento',
                                              'wordpress', 'Javascript', 'Angular JS', 'c#', 'Flask', 'SDK']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                       text='Recommended skills generated from System',
                                                       value=recommended_skills, key='3')
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(web_course)
                        break

                    # Android App Development
                    elif i.lower() in android_keyword:
                        skills_match_score += 10
                        reco_field = 'Android Development'
                        st.success("** Our analysis says you are looking for Android App Development Jobs **")
                        recommended_skills = ['Android', 'Android development', 'Flutter', 'Kotlin', 'XML', 'Java',
                                              'Kivy', 'GIT', 'SDK', 'SQLite']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                       text='Recommended skills generated from System',
                                                       value=recommended_skills, key='4')
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(android_course)
                        break

                    # IOS App Development
                    elif i.lower() in ios_keyword:
                        skills_match_score += 10
                        reco_field = 'IOS Development'
                        st.success("** Our analysis says you are looking for IOS App Development Jobs **")
                        recommended_skills = ['IOS', 'IOS Development', 'Swift', 'Cocoa', 'Cocoa Touch', 'Xcode',
                                              'Objective-C', 'SQLite', 'Plist', 'StoreKit', "UI-Kit", 'AV Foundation',
                                              'Auto-Layout']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                       text='Recommended skills generated from System',
                                                       value=recommended_skills, key='5')
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(ios_course)
                        break

                    # Ui-UX Recommendation
                    elif i.lower() in uiux_keyword:
                        skills_match_score += 10
                        reco_field = 'UI-UX Development'
                        st.success("** Our analysis says you are looking for UI-UX Development Jobs **")
                        recommended_skills = ['UI', 'User Experience', 'Adobe XD', 'Figma', 'Zeplin', 'Balsamiq',
                                              'Prototyping', 'Wireframes', 'Storyframes', 'Adobe Photoshop', 'Editing',
                                              'Illustrator', 'After Effects', 'Premier Pro', 'Indesign', 'Wireframe',
                                              'Solid', 'Grasp', 'User Research']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                       text='Recommended skills generated from System',
                                                       value=recommended_skills, key='6')
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(uiux_course)
                        break

                # Generate timestamp
                ts = time.time()
                timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

                # Resume scoring
                st.subheader("**Resume Tips & Ideas💡**")
                resume_score = 0
                
                # Content checks
                if 'Objective' in resume_text:
                    resume_score += 20
                    st.markdown(
                        '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Objective</h4>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add your career objective, it will give your career intension to the Recruiters.</h4>''',
                        unsafe_allow_html=True)

                if 'Declaration' in resume_text:
                    resume_score += 20
                    st.markdown(
                        '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Declaration✍</h4>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Declaration✍. It will give the assurance that everything written on your resume is true and fully acknowledged by you</h4>''',
                        unsafe_allow_html=True)

                if 'Hobbies' in resume_text or 'Interests' in resume_text:
                    resume_score += 20
                    st.markdown(
                        '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies⚽</h4>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Hobbies⚽. It will show your persnality to the Recruiters and give the assurance that you are fit for this role or not.</h4>''',
                        unsafe_allow_html=True)

                if 'Achievements' in resume_text:
                    resume_score += 20
                    st.markdown(
                        '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Achievements🏅 </h4>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Achievements🏅. It will show that you are capable for the required position.</h4>''',
                        unsafe_allow_html=True)

                if 'Projects' in resume_text:
                    resume_score += 20
                    st.markdown(
                        '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects👨‍💻 </h4>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Projects👨‍💻. It will show that you have done work related the required position or not.</h4>''',
                        unsafe_allow_html=True)

                # Calculate total score using our enhanced algorithm
                total_score = ResumeRanker.calculate_resume_score(
                    resume_data, resume_text, skills_match_score)

                # Display resume score
                st.subheader("**Resume Score📝**")
                st.markdown(
                    """
                    <style>
                        .stProgress > div > div > div > div {
                            background-color: #d73b5c;
                        }
                    </style>""",
                    unsafe_allow_html=True,
                )
                my_bar = st.progress(0)
                score = 0
                for percent_complete in range(resume_score):
                    score += 1
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1)
                st.success('** Your Resume Writing Score: ' + str(score) + '**')
                st.warning(
                    "** Note: This score is calculated based on the content that you have added in your Resume. **")
                
                # Show advanced analytics
                st.subheader("**Advanced Analytics**")
                st.write(f"Total Comprehensive Score: {total_score}")
                st.write(f"Skills Match Score: {skills_match_score}")
                st.write(f"Experience Level: {cand_level}")
                
                st.balloons()

                # Insert data into database
                insert_data(cursor, connection, resume_data['name'], resume_data['email'], 
                          str(resume_score), timestamp, str(resume_data['no_of_pages']), 
                          reco_field, cand_level, str(resume_data['skills']),
                          str(recommended_skills), str(rec_course), total_score, False)

                # Resume writing video
                st.header("**Bonus Video for Resume Writing Tips💡**")
                resume_vid = random.choice(resume_videos)
                res_vid_title = fetch_yt_video(resume_vid)
                st.subheader("✅ **" + res_vid_title + "**")
                st.video(resume_vid)

                # Interview Preparation Video
                st.header("**Bonus Video for Interview👨‍💼 Tips💡**")
                interview_vid = random.choice(interview_videos)
                int_vid_title = fetch_yt_video(interview_vid)
                st.subheader("✅ **" + int_vid_title + "**")
                st.video(interview_vid)

            else:
                st.error('Something went wrong with the resume parsing...')
    else:
        # Admin Side
        
        st.success('Welcome to Admin Side')
        
        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')
        if st.button('Login'):
            if ad_user == 'machine_learning_hub' and ad_password == 'mlhub123':
                st.success("Welcome Admin")
                
                # Admin tabs for different functionalities
                admin_tabs = st.tabs(["User Data", "Best Resumes Selection", "Analytics"])
                
                with admin_tabs[0]:
                    # Display all user data
                    # HERE IS THE FIX FOR THE COLUMN MISMATCH ERROR
                    try:
                        # First, check what columns actually exist in the database
                        cursor.execute("DESCRIBE user_data")
                        db_columns = [column[0] for column in cursor.fetchall()]
                        print(f"Database columns: {db_columns}")
                        
                        # Execute the query with explicit column selection
                        cursor.execute('''
                        SELECT ID, Name, Email_ID, resume_score, Timestamp, Page_no,
                               Predicted_Field, User_level, Actual_skills, Recommended_skills,
                               Recommended_courses
                        FROM user_data
                        ''')
                        
                        data = cursor.fetchall()
                        st.header("**User's👨‍💻 Data**")
                        
                        # Create DataFrame with only the columns we have
                        df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Page',
                                                       'Predicted Field', 'User Level', 'Actual Skills', 'Recommended Skills',
                                                       'Recommended Course'])
                        
                        # Check if total_score and selected columns exist in the database
                        if 'total_score' in db_columns and 'selected' in db_columns:
                            # If they exist, add them to the query
                            cursor.execute('''
                            SELECT total_score, selected FROM user_data
                            ''')
                            additional_data = cursor.fetchall()
                            
                            # Add these columns to the DataFrame
                            if len(additional_data) == len(data):
                                df['Total Score'] = [row[0] for row in additional_data]
                                df['Selected'] = [row[1] for row in additional_data]
                        
                        st.dataframe(df)
                        st.markdown(get_table_download_link(df, 'User_Data.csv', 'Download Report'), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error displaying user data: {e}")
                        # Fallback to simpler query if the above fails
                        cursor.execute('''SELECT * FROM user_data''')
                        data = cursor.fetchall()
                        # Get column names directly from cursor description
                        columns = [column[0] for column in cursor.description]
                        df = pd.DataFrame(data, columns=columns)
                        st.dataframe(df)
                
                with admin_tabs[1]:
                    # Best resumes selection using DAA
                    st.header("**Best Resumes Selection**")
                    
                    # Sorting algorithm selection
                    sorting_algorithm = st.selectbox(
                        "Choose Sorting Algorithm", 
                        ["merge_sort", "quick_sort", "heap_sort"]
                    )
                    
                    # Number of resumes to select
                    num_resumes = st.slider("Number of Resumes to Select", 1, 20, 5)
                    
                    # Fetch all resumes from database
                    resumes = get_all_resumes(cursor)
                    
                    if st.button("Sort Resumes"):
                        if resumes:
                            # Sort resumes using selected algorithm
                            st.write(f"Sorting {len(resumes)} resumes using {sorting_algorithm}...")
                            
                            sorted_resumes = ResumeRanker.select_best_resumes(resumes, algorithm=sorting_algorithm)
                            
                            # Display sorted resumes
                            st.subheader("Sorted Resumes (Highest Score First)")
                            sorted_df = pd.DataFrame([
                                {
                                    'ID': r['id'], 
                                    'Name': r['name'], 
                                    'Email': r['email'], 
                                    'Field': r['predicted_field'],
                                    'Level': r['experience_level'],
                                    'Score': r['total_score']
                                } for r in sorted_resumes
                            ])
                            st.dataframe(sorted_df)
                        else:
                            st.warning("No resumes found in database.")
                    
                    if st.button("Select Optimal Resumes (Knapsack)"):
                        if resumes:
                            # Use knapsack to select optimal combination of resumes
                            st.write(f"Selecting optimal {num_resumes} resumes using Knapsack algorithm...")
                            
                            # First sort by score
                            sorted_resumes = ResumeRanker.select_best_resumes(resumes, algorithm='merge_sort')
                            
                            # Then use knapsack to select optimal combination
                            selected_resumes = ResumeRanker.select_optimal_resumes(
                                sorted_resumes, max_candidates=num_resumes)
                            
                            # Update database to mark selected resumes
                            selected_ids = [r['id'] for r in selected_resumes]
                            if update_selected_status(cursor, connection, selected_ids):
                                st.success(f"Successfully selected {len(selected_ids)} resumes.")
                            
                            # Display selected resumes
                            st.subheader("Selected Resumes")
                            selected_df = pd.DataFrame([
                                {
                                    'ID': r['id'], 
                                    'Name': r['name'], 
                                    'Email': r['email'], 
                                    'Field': r['predicted_field'],
                                    'Level': r['experience_level'],
                                    'Score': r['total_score']
                                } for r in selected_resumes
                            ])
                            st.dataframe(selected_df)
                            
                            # Show why these resumes were selected
                            st.subheader("Selection Explanation")
                            st.write("""
                            The Knapsack algorithm was used to select an optimal combination of resumes that:
                            1. Maximizes the total score (quality of candidates)
                            2. Ensures diversity in skills and domains
                            3. Balances experience levels
                            
                            This selection provides the best possible team composition based on available candidates.
                            """)
                        else:
                            st.warning("No resumes found in database.")
                
                with admin_tabs[2]:
                    # Analytics
                    st.header("**Analytics Dashboard**")
                    
                    # Load data for analytics
                    query = 'select * from user_data;'
                    plot_data = pd.read_sql(query, connection)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart for predicted field recommendations
                        labels = plot_data.Predicted_Field.unique()
                        values = plot_data.Predicted_Field.value_counts()
                        st.subheader("**Predicted Field Distribution**")
                        fig = px.pie(plot_data, values=values, names=labels, 
                                    title='Predicted Field according to Skills')
                        st.plotly_chart(fig)
                    
                    with col2:
                        # Pie chart for User's Experience Level
                        labels = plot_data.User_level.unique()
                        values = plot_data.User_level.value_counts()
                        st.subheader("**Experience Level Distribution**")
                        fig = px.pie(plot_data, values=values, names=labels, 
                                    title="Users Experience Level")
                        st.plotly_chart(fig)
                    
                    # Score distribution histogram
                    if 'total_score' in plot_data.columns:
                        st.subheader("**Score Distribution**")
                        fig = px.histogram(plot_data, x='total_score', 
                                          title='Distribution of Total Scores',
                                          nbins=10)
                        st.plotly_chart(fig)
                    
                    # Selection analytics
                    if 'selected' in plot_data.columns:
                        selected_count = plot_data['selected'].sum()
                        total_count = len(plot_data)
                        
                        st.subheader("**Selection Statistics**")
                        st.write(f"Total Resumes: {total_count}")
                        st.write(f"Selected Resumes: {selected_count}")
                        st.write(f"Selection Rate: {selected_count/total_count:.2%}")
                        
                        # Compare selected vs non-selected
                        if selected_count > 0 and 'total_score' in plot_data.columns:
                            st.subheader("**Selected vs Non-Selected Comparison**")
                            avg_score_selected = plot_data[plot_data['selected'] == True]['total_score'].mean()
                            avg_score_nonselected = plot_data[plot_data['selected'] == False]['total_score'].mean()
                            
                            comparison_data = pd.DataFrame({
                                'Category': ['Selected', 'Non-Selected'],
                                'Average Score': [avg_score_selected, avg_score_nonselected]
                            })
                            
                            fig = px.bar(comparison_data, x='Category', y='Average Score',
                                        title='Average Score Comparison')
                            st.plotly_chart(fig)
            else:
                st.error("Wrong ID & Password Provided")

if __name__ == "__main__":
    run()