import pandas as pd

def Convert(csv_file):
	# Load CSV file using pandas
	df = pd.read_csv(csv_file)
	df = Preproccess(df)

	# Get column names
	columns = df.columns.tolist()

	headerCode = "\hline\n"
	headerCode += "\t\multicolumn{1}{|c|}{\\textbf{"
	headerCode += "}} &\n\t\multicolumn{1}{c|}{\\textbf{".join(columns)

	headerCode += "}} \\\\\n\hline\n"


	# Generate LaTeX table code
	latex_code = "\\begin{longtable}{|" + "c|" * len(columns) + "}\n"

	# add caption and label
	latex_code += "\\caption{Insert Caption Here.}\n"
	latex_code += "\\label{tab:InsertLabelHere} \\\\\n"

	latex_code += headerCode
	latex_code += "\endfirsthead\n\n"

	latex_code += "\multicolumn{" + str(len(columns)) + "}{c}%\n"
	latex_code += "{{\\bfseries \\tablename\\ \\thetable{} -- continued from previous page}} \\\\\n"
	latex_code += headerCode
	latex_code += "\endhead\n\n"

	latex_code += "\hline \multicolumn{" + str(len(columns)) + "}{|c|}{{Continued on next page}} \\\\ \hline\n\n"
	latex_code += "\endfoot\n"

	latex_code += "\hline\n"
	latex_code += "\endlastfoot\n"

	latex_code += "\n"


	# Add data rows
	for index, row in df.iterrows():
		values = row.tolist()
		latex_code += "\t" + " & ".join(str(value) for value in values) + " \\\\\n"
	latex_code += "\\hline\n"
	# Complete LaTeX table code
	latex_code += "\\end{longtable}"

	return latex_code

def Preproccess(df):
	# ask users which columns to drop columns one by one
	columns = df.columns.tolist()
	for column in columns:
		choice = input(f"Drop {column}? (y/n):")
		if choice == "y":
			df = df.drop(column, axis=1)

	print()
	# remove rows with NaN values
	# choice what columns to remove nan values from
	columns = df.columns.tolist()
	for column in columns:
		choice = input(f"Drop NaN values from {column}? (y/n):")
		if choice == "y":
			df = df.dropna(subset=[column])

	print()
	# convert all nans to empty strings
	df = df.fillna("")
	return df

# Provide the path to your CSV file
csv_file_path = "input.csv"
outputPath = "output.tex"

# Convert CSV to LaTeX table code
latex_table = Convert(csv_file_path)

# Print the generated LaTeX code
# print(latex_table)

# Write generated LaTeX code to file
with open(outputPath, "w") as file:
	file.write(latex_table)

print(f"Saved to {outputPath}")