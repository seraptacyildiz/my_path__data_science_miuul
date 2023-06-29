###############################################
# Python Alıştırmalar
###############################################

###############################################
# GÖREV 1: Veri yapılarının tipleriniz inceleyiniz.
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

#py 3.7 sürümünden sonra sıralı özelliği kazandı
# Değiştirilebilir
# Kapsayıcı
# Sırasız
# Key değerleri farklı olacak


t = ("Machine Learning", "Data Science")
type(t)

# Değiştirilemez
# Kapsayıcı
# Sıralı


s = {"Python", "Machine Learning", "Data Science","Python"}
type(s)
s = {"Python", "ML", "Data Science","Python"}

# Değiştirilebilir
# Sırasız + Eşsiz
# Kapsayıcı



###############################################
# GÖREV 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.
###############################################

###############################################
# GÖREV 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.
###############################################

text = "The goal is to turn data into information, and information into insight."
text.upper().replace(","," ").replace("."," ").split()

z = text.upper()
x = z.replace(",", " ")
p = x.replace(".", " ")
w = p.split()
w


###############################################
# GÖREV 3: Verilen liste için aşağıdaki görevleri yapınız.
###############################################

lst = ["D","A","T","A","S","C","I","E","N","C","E"]

# Adım 1: Verilen listenin eleman sayısına bakın.
len(lst)

# Adım 2: Sıfırıncı ve onuncu index'teki elemanları çağırın.
lst[0]
lst[10]

# Adım 3: Verilen liste üzerinden ["D","A","T","A"] listesi oluşturun. "slicing"

data_list = lst[0:4]
data_list

# Adım 4: Sekizinci index'teki elemanı silin.

del lst[8]
lst.pop(8)

lst
lst.remove("N")
list = list[1:]
lst[8] = "N"
# Adım 5: Yeni bir eleman ekleyin.

lst.append(101)
lst


# Adım 6: Sekizinci index'e  "N" elemanını tekrar ekleyin.

lst.insert(8, "N")
lst


###############################################
# GÖREV 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
###############################################

dict = {'Christian': ["America",18],
        'Daisy':["England",12],
        'Antonio':["Spain",22],
        'Dante':["Italy",25]}


# Adım 1: Key değerlerine erişiniz.

dict.keys()

# Adım 2: Value'lara erişiniz.

dict.values()

# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dict.update({"Daisy": ["England",13]})
dict

dict['Daisy'] = ["England",13]

dict["Daisy"][1] = 14
dict["Daisy"][0]="UK"
dict


# Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.

dict.update({"Ahmet": ["Turkey", 24]})
dict

# Adım 5: Antonio'yu dictionary'den siliniz.

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
# GÖREV 5: Arguman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atıyan ve bu listeleri return eden fonskiyon yazınız.
###############################################

l = [2,13,18,93,22]

def func(list):

    çift_list = []
    tek_list = []

    for i in list:
        if i % 2 == 0:
            çift_list.append(i)
        else:
            tek_list.append(i)

    return çift_list, tek_list


çift,tek = func(l)

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

def ayir(liste):
    cift = []
    tek = []
    [cift.append(i) if i%2==0 else tek.append(i) for i in liste]
    return(tek, cift)

ayir(l)


###############################################
# GÖREV 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri bulunmaktadır.
# Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci de tıp fakültesi öğrenci sırasına aittir.
# Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.
###############################################

ogrenciler = ["Ali","Veli","Ayşe","Talat","Zeynep","Ece"]


for i,x in enumerate(ogrenciler):
    if i<3:
        i += 1
        print("Mühendislik Fakültesi",i,". öğrenci: ",x)
    else:
        i -= 2
        print("Tıp Fakültesi",i,". öğrenci: ",x)

for i,x in enumerate(ogrenciler):
    i -= 2
    print(i,x)


for index, ogrenci in enumerate(ogrenciler, 1):
    if index < 4:
        print("Mühendislik Fakültesi", index, ". öğrenci:", ogrenci)
    else:
        print("Tıp Fakültesi", index-3, ". öğrenci:", ogrenci)

a = ogrenciler[0:3]
b = ogrenciler[3:]

for index, ogrenci in enumerate(a, 1):
    print(f"Muhendislik Fakultesi {index}. ogrenci: {ogrenci}")
for index, ogrenci in enumerate(b, 1):
    print(f"Tıp Fakultesi {index}. ogrenci: {ogrenci}")

for i, ogrenci in enumerate(ogrenciler, 1):
    if i <= 3:
        print("Mühendislik Fakültesi", i, ". öğrenci:", ogrenci)
    else:
        print("Tıp Fakültesi", i-3, ". öğrenci:", ogrenci)

Mühendislik_Fakültesi=[]
Tıp_Fakültesi=[]
fakulte=[[],[]]
for i,x in enumerate(ogrenciler,1):
    if i<4:
        fakulte[0].append(x)
        print(f"Mühendislik Fakültesi öğrencisi {i}: {x}")
    else:
        fakulte[1].append(x)
        print(f"Tıp Fakültesi öğrencisi {i-3}: {x}")

for i, ogrenci in enumerate(ogrenciler):
    if i < 3:
        print(f"Mühendislik Fakültesi {i+1}. öğrencisi: {ogrenci}")
    else:
        print(f"Tıp Fakültesi {i-2}. öğrencisi: {ogrenci}")

###############################################
# GÖREV 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yer almaktadır. Zip kullanarak ders bilgilerini bastırınız.
###############################################

ders_kodu = ["CMP1005","PSY1001","HUK1005","SEN2204"]
kredi = [3,4,2,4]
kontenjan = [30,75,150,25]

for ders_kodu, kredi, kontenjan in zip(ders_kodu, kredi, kontenjan):
  print(f"Kredisi {kredi} olan {ders_kodu} kodlu dersin kontenjanı {kontenjan} kişidir.")


###############################################
# GÖREV 8: Aşağıda 2 adet set verilmiştir.
# Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.
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

