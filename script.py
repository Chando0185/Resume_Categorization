import os
import pandas as pd
import pickle
import shutil
from pypdf import PdfReader
import re
import argparse
 
import argparse

parser = argparse.ArgumentParser(description='Command Line Script')
parser.add_argument('--input', help='the input directory', required=True)
parser.add_argument('--output', help='the output directory', required=True)
args = parser.parse_args()


word_vector = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("clf.pkl", "rb"))


def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}
# Command line script to categorize resumes
def categorize_resumes(input_directory, output_directory, output_csv):
    if not os.path.exists(input_directory):
        os.makedirs(input_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    results = []
    if(len(os.listdir(input_directory))==0):
        print("No File Found..")
    else:
        for resume_file in os.listdir(input_directory):
            if resume_file.endswith('.pdf'):  # Change the extension as needed
                resume_path = os.path.join(input_directory, resume_file)
                reader = PdfReader(resume_path)
                page = reader.pages[0]
                text = page.extract_text()
                cleaned_resume = cleanResume(text)

                input_features = word_vector.transform([cleaned_resume])
                prediction_id = model.predict(input_features)[0]
                category_name = category_mapping.get(prediction_id, "Unknown")
                
                category_folder = os.path.join(output_directory, category_name)
                
                if not os.path.exists(category_folder):
                    os.makedirs(category_folder)
                
                target_path = os.path.join(category_folder, resume_file)
                shutil.copy(resume_path, target_path)
                
                results.append({'filename': resume_file, 'category': category_name})
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    input_directory = args.input
    output_directory = args.output
    output_csv = "categorized_resumes.csv"
    
    categorize_resumes(input_directory, output_directory, output_csv)
    print("Resumes categorization and processing completed.")
