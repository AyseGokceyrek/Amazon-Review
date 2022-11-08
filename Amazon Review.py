#############################################
# İş Problemi / Business Problem
#############################################
# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması
# ve satın alanlar için sorunsuz bir alışveriş deneyimi demektir.
# Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması olarak karşımıza çıkmaktadır.
# Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
#  hem de müşteri kaybına neden olacaktır.
# Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler ise
#  satınalma yolculuğunu sorunsuz olarak tamamlayacaktır.


#############################################
# Veri Seti Hikayesi
#############################################
# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.


# 12 Değişken   4915 Gözlem 71.9 MB
# reviewerID:         Kullanıcı ID’si
# asin:               Ürün ID’si
# reviewerName:       Kullanıcı Adı
# helpful:            Faydalı değerlendirme derecesi
# review:             Text Değerlendirme
# overall:            Ürün rating’i
# summary:            Değerlendirme özeti
# unixReviewTime:     Değerlendirme zamanı
# reviewTime:         Değerlendirme zamanı Raw
# day_diff:           Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes:        Değerlendirmenin faydalı bulunma sayısı
# total_vote:         Değerlendirmeye verilen oy sayısı

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


#############################################
# PROJE GÖREVLERİ / PROJECT TASKS
#############################################

df = pd.read_csv("C:/Users/Lenovo/PycharmProjects/datasets/amazon_review.csv")
df.head()

#############################################
# GÖREV 1: Average Rating’i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.
#############################################
# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.

# Adım 1:   Ürünün ortalama puanını hesaplayınız.

df.describe().T
df["overall"].mean()


# Adım 2: Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.

# reviewTime değişkenini tarih değişkeni olarak tanıtmanız
# reviewTime'ın max değerini current_date olarak kabul etmeniz
# her bir puan-yorum tarihi ile current_date'in farkını gün cinsinden ifade ederek yeni değişken oluşturmanız ve
# gün cinsinden ifade edilen değişkeni quantile fonksiyonu ile 4'e bölüp (3 çeyrek verilirse 4 parça çıkar)
# çeyrekliklerden gelen değerlere göre ağırlıklandırma yapmanız gerekir.
# Örneğin q1 = 12 ise ağırlıklandırırken 12 günden az süre önce yapılan yorumların ortalamasını alıp
# bunlara yüksek ağırlık vermek gibi.


df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date = pd.to_datetime(str(df["reviewTime"].max()))
df["day_"] = (current_date - df["reviewTime"]).dt.days
df.head()

Q1, Q2, Q3 = df["day_"].quantile([.25, .5, .75])


def time_based_weighted_average(dataframe, w1=35, w2=30, w3=20, w4=15):
    return dataframe.loc[dataframe["day_"] <= Q1, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_"] > Q1) & (dataframe["day_diff"] <= Q2), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_"] > Q2) & (dataframe["day_diff"] <= Q3), "overall"].mean() * w3 / 100 + \
           dataframe.loc[dataframe["day_"] > Q3, "overall"].mean() * w4 / 100


# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.

print(f"Average score between 0-280 days: ", df[df["day_"] <= Q1]["overall"].mean())
print(f"Average score between 281-430 days: ", df[(df["day_"] > Q1) & (df["day_"] <= Q2)]["overall"].mean())
print(f"Average score between 431-600 days: ", df[(df["day_"] > Q2) & (df["day_"] <= Q3)]["overall"].mean())
print(f"Average score over 601 days: ", df[df["day_"] > Q3]["overall"].mean())

"""
Yukarıdaki karşılaştırmada verisetimizdeki günler çeyreklik dilimlere bölünmüş olup, 280'den küçük olan ortalamaya 
1. çeyrek, 280-430'a 2. çeyrek, 430-600 arasında 3. çeyrek ve 600'den büyük günlere de 4. çeyrek şeklinde yorumlanacaktır.
Burada 1. çeyreklikteki günlerimiz en yakın günler olduğu için bizim için değerlidir ve 35 puan olarak ağırlıklandırılmıştır.
Günler ilerledikçe ya da arttıkça verilen oylardaki önem azalacağı için ağırlıklandırmalar da buna göre verilmiştir.

Dolayısı ile yukarıdaki işlemler sonucu çıkan değerleri gözlemlediğimizde; günümüze yaklaştıkça verilen 
puan ortalamalarında artış olmuştur.

"""

###############################################################
# GÖREV 2. Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.
###############################################################
# Adım 1: helpful_no değişkenini üretiniz.

# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# Veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.
# Toplam oy sayısından (total_vote) yararlı oy sayısı (helpful_yes) çıkarılarak
# yararlı bulunmayan oy sayılarını (helpful_no) bulunuz.

df["helpful_no"] = (df["total_vote"] - df["helpful_yes"])

df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "total_vote", "reviewTime"]]
df.head()
# Adım 2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz.

# score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayabilmek için score_pos_neg_diff,
# score_average_rating ve wilson_lower_bound fonksiyonlarını tanımlayınız.
# score_pos_neg_diff'a göre skorlar oluşturunuz. Ardından; df içerisinde score_pos_neg_diff ismiyle kaydediniz.
# score_average_rating'a göre skorlar oluşturunuz. Ardından; df içerisinde score_average_rating ismiyle kaydediniz.
# wilson_lower_bound'a göre skorlar oluşturunuz. Ardından; df içerisinde wilson_lower_bound ismiyle kaydediniz.



def score_up_down_diff(up, down):
    return up - down


df['score_pos_neg_diff'] = df.apply(lambda x: score_up_down_diff(x['helpful_yes'], x['helpful_no']), axis=1)


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


df['score_average_rating'] = df.apply(lambda x: score_average_rating(x['helpful_yes'], x['helpful_no']), axis=1)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x['helpful_yes'], x['helpful_no']), axis=1)

df.head(20)

# Adım 3: 20 Yorumu belirleyiniz ve sonuçları Yorumlayınız.
# wilson_lower_bound'a göre ilk 20 yorumu belirleyip sıralayanız.
# Sonuçları yorumlayınız.

df.sort_values("score_pos_neg_diff", ascending=False).head(20)
df.sort_values("score_average_rating", ascending=False).head(20)
df.sort_values("wilson_lower_bound", ascending=False).head(20)





