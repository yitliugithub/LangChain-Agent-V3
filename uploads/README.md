# User Uploads

Put completed brand research `.xlsx` files in this directory.

From the project root, start the Agent and pass the uploaded workbook path:

```text
python app_http.py
User: report uploads/research_input_template.xlsx
```

The generated `research_brief.json` is an internal intermediate file and stays
next to the workbook for traceability.
