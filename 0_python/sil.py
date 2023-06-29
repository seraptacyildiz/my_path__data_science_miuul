Dengesiz Veri Seti
Bir iş için sınıflandırma modeli yaratıyorsunuz ve bu modelin doğruluk değeri %95 çıkıyor. Her şey güzel. Modeliniz kullanılmaya başlanıyor ama başarısız olduğu, her zaman aynı sınıfı tahminlediği fark ediliyor. Veri setini tekrar incelediğinizde tespit edilemeyen sınıfın veri setindeki oranın %5 olduğunu görüyorsunuz. Yani modelinizin başarısızlığı dengesiz veri setinden (Imbalanced Dataset) kaynaklı.

Dengesiz veri seti sınıflandırma problemlerinde görülür ve sınıf dağılımlarının birbirine yakın olmadığı durumlarda ortaya çıkar. Problem çoğunluğa sahip sınıfın azınlık sınıfını domine etmesinden kaynaklanır. Oluşturulan model çoğunluğa sahip sınıfa yakınlık gösterir, bu da azınlık sınıfının kötü sınıflandırılmasına sebep olur.

Dengesiz veri setleriyle karşılaştığımızda doğru gözlem yapabilmek ve dengeyi sağlayabilmek için uygulayabileceğimiz çeşitli yöntemler vardır:

Doğru Metrik Seçimi
Precision
Recall
F1-score
ROC Curve
AUC
Resampling
Oversampling
Random Oversampling
SMOTE Oversampling
Undersampling
Random Undersampling
NearMiss Undersampling
Undersampling (Tomek links)
Undersampling (Cluster Centroids)
Daha fazla veri toplamak
Sınıflandırma modellerinde bulunan “class_weight” parametresi kullanılarak azınlık ve çoğunluk sınıflarından eşit şekilde öğrenebilen model yaratılması,
Tek bir modele değil , diğer modellerdeki performanslara da bakılması,
Daha farklı bir yaklaşım uygulanıp Anomaly detection veya Change detection yapmak
Dengesizlik içeren Credit Card Fraud Detection veri setini inceleyip, daha sonrasında bu dengesizlikle başa çıkabilmek için veri setine çeşitli yöntemler uygulayacağız.

Veri Setinin İncelenmesi
Kredi kartı şirketlerinin dolandırıcılığı tespit etmeleri önemlidir, müşterilerine yanlış ücretlendirilme yapılmasını istemezler.

Bu veri seti kullanılarak dolandırıcılık yapılmış kredi kartlarını tespit eden bir model oluşturulmak isteniliyor.

Veri seti eylül 2013'te avrupada kredi kartıyla yapılan işlemlerden ve bu işlemlerin fraud(dolandırıcılık) ise 1 değilse 0 olarak etiketlenmesiyle oluşmuştur. Gizlilik nedeniyle arka plan bilgisi çok fazla bulunmuyor ve "Time","Amount" değişkeni haricinde diğer değişkenler PCA(Principal component analysis) ile dönüştürülmüştür.

"Time" : ilk işlem ile her işlem arasındaki saniye
"Amount": maliyet
ilk olarak veri setini okutup, boş değer olup olmadığını gözlemleyip, Class değişkenin dağılımına bakacağız. Daha sonrasında "Amount" ve "Time" değişkenini standartlaştırıyoruz. Veriyi hold out yöntemiyle ayırıp, sınıflandırma modeli olan logistic regression ile modeli oluşturuyoruz. Tek bir model üzerinden gideceğiz.
