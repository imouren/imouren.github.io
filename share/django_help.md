
## django上传excel

```python

@csrf_exempt
def test_excel(request):
    input_excel = request.FILES.get("excel_file")
    data = xlrd.open_workbook(file_contents=input_excel.read())
    table = data.sheet_by_index(0)
    nrows = table.nrows
    # ncols = table.ncols
    colnames = table.row_values(0)
    alist = []
    for rownum in range(1, nrows):
        row = table.row_values(rownum)
        if row:
            app = {}
            for i in range(len(colnames)):
                app[colnames[i]] = row[i]
            alist.append(app)
    print alist
    print colnames
    return HttpResponse("ok")
```

测试html

```python

<html>
    <head>
        <title>
            Excel file upload and processing : Django Example : ThePythonDjango.Com
        </title>
    </head>
    <body style="margin-top: 30px;margin-left: 30px;">
        <form action="http://localhost:5500/ecommerce/test/excel/" method="post" enctype="multipart/form-data">
            <input type="file"
                   title="Upload excel file"
                   name="excel_file"
                   style="border: 1px solid black; padding: 5px;"
                   required="required">
            <p>
            <input type="submit"
                   value="Upload"
                   style="border: 1px solid green; padding:5px; border-radius: 2px; cursor: pointer;">
        </form>

        <p></p>
        <hr>
    </body>
</html>

```
