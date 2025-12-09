import re
string =input()
# 任务：采用re库函数，对string分别进行密码格式认证提取，
#验证输入密码符合要求（8位以上，字母开头，只能是字母、数字、下划线）
# ********** Begin *********#

password_pattern=r"\b[a-zA-Z0-9_]+\b"
passwords = re.findall(password_pattern,string)

print("提取密码是")
for password in passwords:
    print(password)

# ********** End **********#