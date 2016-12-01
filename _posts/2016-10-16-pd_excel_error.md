---
layout: post
title:  "pandas导出excel错误"
date:   2016-10-16 09:05:31
categories: python pandas
tags: python
---

* content
{:toc}


出错信息

```python
IllegalCharacterError                     Traceback (most recent call last)
<ipython-input-20-6fab861fcc4e> in <module>()
----> 1 save_to_excel(res, video_info, output_path)

<ipython-input-19-bc52842edc55> in save_to_excel(res, info_dict, output_path)
     19     local_path = "/tmp/" + output_path
     20     writer = pd.ExcelWriter(local_path)
---> 21     output.to_excel(writer, sheet_name="info")
     22     writer.save()

/opt/workon_home/gotyou2/lib/python2.7/site-packages/pandas/core/frame.pyc in to_excel(self, excel_writer, sheet_name, na_rep, float_format, columns, header, index, index_label, startrow, startcol, engine, merge_cells, encoding, inf_rep, verbose)
   1460         formatted_cells = formatter.get_formatted_cells()
   1461         excel_writer.write_cells(formatted_cells, sheet_name,
-> 1462                                  startrow=startrow, startcol=startcol)
   1463         if need_save:
   1464             excel_writer.save()

/opt/workon_home/gotyou2/lib/python2.7/site-packages/pandas/io/excel.pyc in write_cells(self, cells, sheet_name, startrow, startcol)
   1313                 column=startcol + cell.col + 1
   1314             )
-> 1315             xcell.value = _conv_value(cell.val)
   1316 
   1317             style_kwargs = {}

/opt/workon_home/gotyou2/lib/python2.7/site-packages/openpyxl/cell/cell.pyc in value(self, value)
    289     def value(self, value):
    290         """Set the value and infer type and display options."""
--> 291         self._bind_value(value)
    292 
    293     @property

/opt/workon_home/gotyou2/lib/python2.7/site-packages/openpyxl/cell/cell.pyc in _bind_value(self, value)
    188 
    189         elif isinstance(value, STRING_TYPES):
--> 190             value = self.check_string(value)
    191             self.data_type = self.TYPE_STRING
    192             if len(value) > 1 and value.startswith("="):

/opt/workon_home/gotyou2/lib/python2.7/site-packages/openpyxl/cell/cell.pyc in check_string(self, value)
    153         value = value[:32767]
    154         if next(ILLEGAL_CHARACTERS_RE.finditer(value), None):
--> 155             raise IllegalCharacterError
    156         return value
    157 

IllegalCharacterError: 
```

从错误信息找代码出处

```python

value = value[:32767]
if next(ILLEGAL_CHARACTERS_RE.finditer(value), None):
    raise IllegalCharacterError
return value

```

意思很明显了, 如果找非非法字符则抛出错误，ILLEGAL_CHARACTERS_RE 就定义了非法字符

```python
ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
```

处理 name 时用这个正则去掉非法字符即可

```python
name = ILLEGAL_CHARACTERS_RE.sub(r'', name)
```
