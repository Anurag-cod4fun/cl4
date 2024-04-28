# Develop a MapReduce program to find the grades of student’s.

# python grade_analyzer.py grades.txt > average_grades.txt
# should have grades.txt

from mrjob.job import MRJob

class GradeAnalyzer(MRJob):
    def mapper(self, _, line):
        # Split the line into student name and grade
        student, grade = line.split(',')
        # Emit the student name as the key and the grade as the value
        yield student.strip(), float(grade.strip())

    def reducer(self, student, grades):
        # Calculate the total grade and count of grades for each student
        total_grade = 0
        count = 0
        for grade in grades:
            total_grade += grade
            count += 1
        # Calculate the average grade for the student
        average_grade = total_grade / count
        # Emit the student name and average grade
        yield student, average_grade

if __name__ == '__main__':
    GradeAnalyzer.run()
