import pandas as pd

df = pd.read_csv('data/student_performance.csv')

print('========DATASET=========')
print(df)

print('\n========DATASET=========')
print(df.info())

print('========MISSING VALUES=========')
print(df.isnull().sum())

print('========DUPLICATES=========')
print(df.duplicated().sum())

print('=========STATISTICS========')
print(df.describe())

print('========AVERAGE=========')
print('Average study hours:',df['hours_studied'].mean())
print('Average attendance:',df['attendance'].mean())
print('Average exam score:',df['exam_score'].mean())

highest = df.loc[df['exam_score'].idxmax()]

print('========TOP STUDENTS=========')
print('Name: ',highest['name'])
print('Exam score: ',highest['exam_score'])

print('========STUDENTS ABOVE 70=========')
print(df[df['exam_score'] > 70])

print('========LOW ATTENDANCE=========')
print(df[df['attendance'] < 80])