# Visura agent implementation
from bs4 import BeautifulSoup

html_table = '''
<table>
  <tr><th><b>Product</b></th><th><span style="color:red">Profit</span></th></tr>
  <tr><td>A</td><td>$100K</td></tr>
</table>
'''

soup = BeautifulSoup(html_table, 'html.parser')
emphasis = [(tag.name, tag.text) for tag in soup.find_all(['b', 'span'])]
print("Visual Cues:", emphasis)
