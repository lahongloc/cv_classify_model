import nltk


# Download necessary NLTK datasets
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        print("NLTK data downloaded successfully.")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")


# download_nltk_data()
from nltk.corpus import wordnet, stopwords
import string
import nltk

# Initialize stop words
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)


# Preprocess text
def preprocess_text(text):
    # Tokenize text, remove stop words and punctuation
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word not in stop_words and word not in punctuation]

    # Return processed text
    return ' '.join(words)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Calculate similarity between two texts
def calculate_similarity(cv_text, job_desc_text):
    # Preprocess the texts
    cv_text_processed = preprocess_text(cv_text)
    job_desc_text_processed = preprocess_text(job_desc_text)

    # Create a CountVectorizer to convert text to vectors
    vectorizer = CountVectorizer().fit_transform([cv_text_processed, job_desc_text_processed])
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(vectors)

    return cosine_sim[0][1]  # Return similarity score between CV and job description


# Example CV and Job Description
cv_text = """LA HONG LOC
FULLSTACK DEVELOPER
0334 454 203 hongloc111990@gmail.com https://github.com/lahongloc Go Vap, Ho Chi Minh City
SUMMARY
• Enthusiastic last-year Information Technology student with a solid understanding of software
development processes and proficiency in programming languages including Python, Java, C#,
HTML, CSS, and Javascript.
• Adept at reading and comprehending technical documentation in English, possess good
communication skills in English.
EDUCATION
• HO CHI MINH CITY OPEN UNIVERSITY(GPA: 3.57/4.0 – 8.39/10) 2021 – 2025
• UDACITY: SQL For Data Analyst 2023 – 2023
• MulticampusVN – SamSung course: Big Data 2024 - Ongoing
SKILLS
• Programming langueges: Java, Javascript, Python, C#
• Frameworks/Libraries/Platforms: Spring boot, NodeJs, ReactJs, Material UI, Python
Flask, Java for mobile development, Boostrap4/5, Cloudinary, Amazon S3, Selenium.
• Databases management system: MySQL, SQL Server, MongoDB.
• Languages: English(ntermediate): Have the ability to read and understand English
documents well, confident in interviewing in English)
PROJECTS
TradingDocs (https://github.com/lahongloc/fatmouseShop) 13/07/2024 – Ongoing
• Project Description: A website designed for the exchange, gifting, or selling of documents,
providing users with a seamless platform to manage their transactions.
• Team Size: Personal project
• My Role and Responsibilities:
o Led the full-stack development of the website.
o Implemented OAuth2 and JWT for secure user authentication and authorization.
o Utilized Redis for caching and session management.
o Managed the database using MongoDB.
o Designed and developed a dynamic and responsive user interface using ReactJs
combined with the MUI library.
• Technologies Used:
o Frontend: ReactJs, MUI library, Animate.css
o Backend: Express.js, Node.js, Redis, OAuth2, JWT, MongoDBCourse Outline Management WebApp
(https://github.com/lahongloc/CourseOutlineManagementWebApp) 15/06/2024 – 8/7/2024
• Description: A webapp used in managing courses outlines for lecturers, students and
administrators.
• Team size: 2 people.
• My responsibilities:
o Configured the server, security, and authorizations with Spring Framework.
o Assisted in designing and developing the client side using ReactJs and the MUI library,
focusing on main features.
o Created and enhanced the Restful API handler for efficient performance.
• Integrated with: Spring mail, VNPay, AWS Amazone S3, Cloudinary, VNPay, Firebase(real-time
chatting), Machine Learning(Sentiment analysis).
• Technologies used:
o Frontend: ReactJs, MUI library, Animate.css.
o Backend: Spring MVC, Spring Security, JPA, Hibernate, Front controller partern, MySQL.
JobBridge (https://github.com/lahongloc/JobBridgeWebApp) 5/8/2024 – Ongoing
• Description: JobBridge is a job-seeking and recruitment website that connects job seekers with
employers. It features a recommendation system using a Naive Bayes classifier to match CVs
with job descriptions.
• Team size: Personal project.
• My responsibilities:
o Developed the backend with Spring Boot and MySQL, test API using Postman.
o Created a responsive frontend using ReactJs and Ant Design.
o Built a recommendation system with Python Flask, integrating machine learning for job
and candidate matching.
• Technologies used:
o Frontend: ReactJs, Ant Design.
o Backend: Spring Boot, Python flask, Naive Bayes Classifier, MySQL.
SKILLSETS
• Office software: Highly proficient in office software.
• UI/UX Design: Figma: Intermediate - Skilled in using Figma for designing and conceptualizing
UI/UX.
• Source Code Management and Project Development: GitHub: Advanced proficiency in utilizing
GitHub for project development and management.
• Integrated Development Environments (IDEs): Extensive experience with various IDEs: Visual
Studio, Visual Studio Code, NetBeans, PyCharm, Anaconda, etc.
• Data Analysis: Competent in data analysis with Python, Excel, SQL, Power BI,.."""
job_desc_text = """Job Title: Full Stack Developer Intern
Location: Ho Chi Minh City
Company Overview:  
[Your Company Name] is a dynamic and innovative tech company dedicated to delivering high-quality software solutions. We are looking for a motivated Full Stack Developer Intern to join our team and assist in the development of exciting projects that shape the future of our industry.
Job Summary:  
As a Full Stack Developer Intern, you will work closely with our development team to design, develop, and implement web applications. You will have the opportunity to learn and apply various technologies while contributing to real-world projects that make a difference.
Key Responsibilities:
Collaborate with team members to develop user-friendly web applications using modern technologies.Assist in the full software development lifecycle, including requirement analysis, design, implementation, testing, and deployment.
Work on both frontend and backend development tasks, ensuring seamless integration and functionality.
Participate in code reviews and contribute to improving coding standards and practices.
Utilize frameworks such as ReactJS for frontend development and Spring Boot for backend development.
Manage databases using MySQL or MongoDB, ensuring data integrity and security.
Implement user authentication and authorization features using OAuth2 and JWT.
Engage in debugging, troubleshooting, and performance optimization of web applications.
Qualifications:
Currently pursuing a Bachelor’s degree in Computer Science, Information Technology, or a related field.
Proficiency in programming languages such as Java, JavaScript, Python, and C#.
Experience with frameworks and libraries, including Spring Boot, ReactJS, and Node.js.
Familiarity with RESTful APIs and database management systems (MySQL, MongoDB).
Basic understanding of UI/UX principles and experience using design tools like Figma.
Good communication skills in English, with the ability to collaborate effectively in a team environment.
Self-motivated and eager to learn new technologies and methodologies.
Preferred Skills:
Understanding of version control systems, particularly Git.
Knowledge of cloud services like AWS or Azure.
Familiarity with machine learning concepts and data analysis.
What We Offer:
An opportunity to gain hands-on experience in a professional setting.
Mentorship from experienced developers and industry professionals.
Exposure to various projects and technologies, enhancing your skill set.
A collaborative and innovative work environment.
How to Apply:  
Interested candidates are invited to submit their CV along with a cover letter detailing their interest in the position and relevant experiences to [Your Email Address].
"""

# Calculate and print similarity
similarity = calculate_similarity(cv_text, job_desc_text)
print("Similarity between CV and Job Description:", similarity * 100)
