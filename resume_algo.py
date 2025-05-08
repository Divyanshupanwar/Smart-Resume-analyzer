# resume_algo.py

def bubble_sort(arr, key=None):
    """
    Simple bubble sort implementation that can sort based on a key function
    
    Args:
        arr: List of items to sort
        key: Optional function to extract a comparison key from each element
    
    Returns:
        Sorted list
    """
    n = len(arr)
    sorted_arr = arr.copy()
    
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            val_j = key(sorted_arr[j]) if key else sorted_arr[j]
            val_j1 = key(sorted_arr[j+1]) if key else sorted_arr[j+1]
            
            if val_j > val_j1:
                sorted_arr[j], sorted_arr[j+1] = sorted_arr[j+1], sorted_arr[j]
                swapped = True
        
        if not swapped:
            break
    
    return sorted_arr

def knapsack_shortlist(candidates, max_candidates=10):
    """
    Uses the 0/1 knapsack algorithm to shortlist optimal candidates
    
    Args:
        candidates: List of dictionaries with candidate info including 'score' and 'skills_match'
        max_candidates: Maximum number of candidates to shortlist
    
    Returns:
        List of shortlisted candidates
    """
    n = len(candidates)
    
    # If we have fewer candidates than the maximum, return all
    if n <= max_candidates:
        return candidates
    
    # Calculate a combined value for each candidate (resume_score + skills match)
    for candidate in candidates:
        candidate['combined_value'] = int(candidate['resume_score']) + candidate['skills_match'] * 20
    
    # Sort candidates by combined value using our sorting algorithm
    sorted_candidates = bubble_sort(candidates, key=lambda x: x['combined_value'])
    sorted_candidates.reverse()  # Sort in descending order
    
    # Use dynamic programming for knapsack to maximize overall value
    # Create a table for dynamic programming
    dp = [[0 for _ in range(max_candidates + 1)] for _ in range(n + 1)]
    selected = [[False for _ in range(max_candidates + 1)] for _ in range(n + 1)]
    
    # Fill the dp table
    for i in range(1, n + 1):
        for j in range(1, max_candidates + 1):
            # We're treating all candidates as weight=1 in this simplified knapsack
            if j >= 1:  # Can include this candidate
                if dp[i-1][j-1] + sorted_candidates[i-1]['combined_value'] > dp[i-1][j]:
                    dp[i][j] = dp[i-1][j-1] + sorted_candidates[i-1]['combined_value']
                    selected[i][j] = True
                else:
                    dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = dp[i-1][j]
    
    # Backtrack to find the selected candidates
    shortlisted = []
    i, j = n, max_candidates
    
    while i > 0 and j > 0:
        if selected[i][j]:
            shortlisted.append(sorted_candidates[i-1])
            j -= 1
        i -= 1
    
    return shortlisted

def merge_sort(arr, key=None):
    """
    Merge sort implementation that can sort based on a key function
    
    Args:
        arr: List of items to sort
        key: Optional function to extract a comparison key from each element
    
    Returns:
        Sorted list
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid], key)
    right = merge_sort(arr[mid:], key)
    
    return merge(left, right, key)

def merge(left, right, key):
    """Helper function for merge sort"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        left_val = key(left[i]) if key else left[i]
        right_val = key(right[j]) if key else right[j]
        
        if left_val <= right_val:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def skill_match_score(resume_skills, job_skills):
    """
    Calculate the skill match score between resume skills and job skills
    
    Args:
        resume_skills: List of skills from the resume
        job_skills: List of skills required for the job
    
    Returns:
        A score between 0 and 1 representing the match
    """
    if not job_skills or not resume_skills:
        return 0
    
    # Convert to lowercase for comparison
    resume_skills_lower = [skill.lower() for skill in resume_skills]
    job_skills_lower = [skill.lower() for skill in job_skills]
    
    # Count matching skills
    matches = sum(1 for skill in resume_skills_lower if skill in job_skills_lower)
    
    # Calculate match percentage
    return matches / len(job_skills_lower)

def rank_candidates_for_job(resumes_data, job_skills):
    """
    Rank candidates for a specific job based on their resume and skill match
    
    Args:
        resumes_data: List of dictionaries with resume data
        job_skills: List of skills required for the job
    
    Returns:
        Ranked list of candidates
    """
    candidates = []
    
    for resume in resumes_data:
        skill_score = skill_match_score(resume['skills'], job_skills)
        candidates.append({
            'name': resume['name'],
            'email': resume['email'],
            'resume_score': resume.get('resume_score', 0),
            'skills_match': skill_score,
            'skills': resume['skills']
        })
    
    # Use our sorting algorithm to sort by combined score
    ranked_candidates = bubble_sort(
        candidates, 
        key=lambda x: int(x['resume_score']) + x['skills_match'] * 20
    )
    
    # Return in descending order
    return list(reversed(ranked_candidates))