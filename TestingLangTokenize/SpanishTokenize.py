import wordsegment
wordsegment.load()

text = "piezasenvolturadecabezademodaslidaminimalistaparamujerparainterior"
tokens = wordsegment.segment(text)

print(tokens)
