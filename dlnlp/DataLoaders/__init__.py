import re
def get_content(path_f):
    fid = open(path_f, "r", encoding='utf8')
    f_list = []
    f_data = fid.read()
    for line in fid.readlines():
         if re.match(r'^\s+$',line):
             continue
         f_list.append(line)
    fid.close()
    f_list = f_data.split(".")
    return(f_list)

