import re
def get_content(path_f):
    fid = open(path_f, "r", encoding='utf8')
    f_list = []
    for line in fid.readlines():
         if re.match(r'^\s+$',line):
             continue
         f_list.append(line)
    fid.close()
    return(f_list)