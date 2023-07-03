###############################################
# Python Exercises
###############################################

###############################################
# ASSIGNMENT 1: Check the types of data structures.
###############################################

x = 8
type(x)

y = 3.2
type(y)

z = 8j + 18
type(z)

a = "Hello World"
type(a)

b = True
type(b)

c = 23 < 22
type(c)


l = [1, 2, 3, 4,"String",3.2, False]
type(l)
l[0]=100
l[-2]
# Sıralıdır
# Kapsayıcıdır
# Değiştirilebilir

d = {"Name": "Jake",
     "Age": [27,56],
     "Adress": "Downtown"}
type(d)

#py Gained the sequential feature after version 3.7
# Mutable
# Be nested
# Unordered
# Key values are different


t = ("Machine Learning", "Data Science")
type(t)

# Unmutable
# Be nested
# Ordered


s = {"Python", "Machine Learning", "Data Science","Python"}
type(s)
s = {"Python", "ML", "Data Science","Python"}

# Mutable
# Unordered + Unique
# Be nested



###############################################
# ASSIGNMENT 2: Convert all letters of the given string to uppercase. Put space instead of commas and periods, and separate them word by word.
###############################################

###############################################
# ASSIGNMENT 2: Convert all letters of the given string to uppercase. Put space instead of commas and periods, and separate them word by word.
###############################################

text = "The goal is to turn data into information, and information into insight."
text.upper().replace(","," ").replace("."," ").split()

z = text.upper()
x = z.replace(",", " ")
p = x.replace(".", " ")
w = p.split()
w


###############################################
# ASSIGNMENT 3: Complete the following tasks for the given list.
###############################################

lst = ["D","A","T","A","S","C","I","E","N","C","E"]

# Step 1: Check the number of elements of the given list.
len(lst)

# Step 2: Call the elements at index zero and ten.
lst[0]
lst[10]

# Step 3: Create a list ["D","A","T","A"] from the given list. "slicing"

data_list = lst[0:4]
data_list

# Step 4: Delete the element in the eighth index.

del lst[8]
lst.pop(8)

lst
lst.remove("N")
list = list[1:]
lst[8] = "N"

# Adım 5: Add a new element.

lst.append(101)
lst


# Step 6: Add the "N" element back to the eighth index.

lst.insert(8, "N")
lst


###############################################
# ASSIGNMENT 4: Apply the following steps to the given dictionary structure.
###############################################

dict = {'Christian': ["America",18],
        'Daisy':["England",12],
        'Antonio':["Spain",22],
        'Dante':["Italy",25]}


# Step 1: Get the Key values.

dict.keys()

# Step 2: Get the Values.

dict.values()

# Step 3: Update the value 12 of the Daisy key to 13.

dict.update({"Daisy": ["England",13]})
dict

dict['Daisy'] = ["England",13]

dict["Daisy"][1] = 14
dict["Daisy"][0]="UK"
dict


# Step 4: Add a new value whose key value is Ahmet value [Turkey,24].

dict.update({"Ahmet": ["Turkey", 24]})
dict

# Step 5: Delete Antonio from dict.

dict.pop("Antonio")
dict

del(dict["Antonio"])

def keys_swap(orig_key, new_key, d):
    d[new_key] = d.pop(orig_key)

keys_swap("Ahmet","Ali",dict)
dict
dict["ali"]=['America', 18]

dict["aaa"] = dict.pop("ali")
###############################################
# ASSIGNMENT 5: Write a function that takes a list as an argument, assigns the odd and even numbers in the list to separate lists, and returns these lists.
###############################################

l = [2,13,18,93,22]

def func(list):

    even_list = []
    odd_list = []

    for i in list:
        if i % 2 == 0:
            even_list.append(i)
        else:
            odd_list.append(i)

    return even_list, odd_list


even, odd = func(l)

def func(l):
    list = [[], []]
    for i in l:
        if i % 2 == 0:
            list[0].append(i)
        else:
            list[1].append(i)
    return(list)

even_list,odd_list = func(l)

def function(x):
    odd_list=[]
    even_list=[]
    for i in x:
        if i % 2 == 0:
            even_list.append(i)
        else:
            odd_list.append(i)
    return even_list, odd_list
function(l)

def separate(liste):
    even = []
    odd = []
    [even.append(i) if i%2==0 else odd.append(i) for i in liste]
    return(odd, even)

separate(l)


###############################################
# ASSIGNMENT 6: In the list given below are the names of the students who received degrees in engineering and medicine faculties.
# While the first three students represent the success order of the engineering faculty, the last three students belong to the medical faculty student rank, respectively.
# Print the student's degrees specific to the faculty using Enumarate.
###############################################

students = ["Ali","Veli","Ayşe","Talat","Zeynep","Ece"]


for i,x in enumerate(students):
    if i<3:
        i += 1
        print("Engineering Faculty",i,". student: ",x)
    else:
        i -= 2
        print("Medicine Faculty",i,". student: ",x)

for i,x in enumerate(student):
    i -= 2
    print(i,x)


for index, student in enumerate(students, 1):
    if index < 4:
        print("Engineering Faculty", index, ". student:", student)
    else:
        print("Medicine Faculty", index-3, ". student:", student)

a = students[0:3]
b = students[3:]

for index, student in enumerate(a, 1):
    print(f"Engineering Faculty {index}. student: {ogrenci}")
for index, ogrenci in enumerate(b, 1):
    print(f"Medicine Faculty {index}. student: {student}")

for i, ogrenci in enumerate(ogrenciler, 1):
    if i <= 3:
        print("Engineering Faculty", i, ". student:", student)
    else:
        print("Medicine Faculty", i-3, ". student:", student)

Engineering_Faculty=[]
Medicine_Faculty=[]
faculty=[[],[]]
for i,x in enumerate(students,1):
    if i<4:
        fakulte[0].append(x)
        print(f"Engineering Faculty students are {i}: {x}")
    else:
        fakulte[1].append(x)
        print(f"Medicine Faculty students are {i-3}: {x}")

for i, student in enumerate(students):
    if i < 3:
        print(f"Engineering Faculty {i+1}. student: {student}")
    else:
        print(f"Medicine Faculty {i-2}. student: {student}")

###############################################
# ASSIGNMENT 7: Three lists are given below. In the lists, there is a course code, credit and quota information, respectively. Print course information using zip.
###############################################

course_code = ["CMP1005","PSY1001","HUK1005","SEN2204"]
credit = [3,4,2,4]
quota = [30,75,150,25]

for course_code, credit, quota in zip(course_code, credit, quota):
  print(f"The quota of the {course_code} coded course with {credit} credits is {quota} people.")



###############################################
# ASSIGNMENT 8: Below are 2 sets.
# You are asked to define the function that will print the difference of the 2nd set from the 1st set, if the 1st set includes the 2nd set, if it does not cover their common elements.
###############################################

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])


def kume(set1,set2):
    if set1.issuperset(set2):
        print(set1.intersection(set2))
    else:
        print(set2.difference(set1))

kume(kume1,kume2)

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

def fonksiyon(kume1,kume2):
    if kume1.issuperset(kume2) == True:
        print(kume1.intersection(kume2))
    else:
        print(kume2.difference(kume1))


fonksiyon(kume1,kume2)

