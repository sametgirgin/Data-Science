
#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.

import pandas as pd
#############################################
# Görev 1: Verinin Hazırlanması
#############################################
# Adım 1: Movie ve Rating veri setlerini okutunuz.

movies = pd.read_csv("HybridRecommender-221114-235254/datasets/movie.csv")
ratings = pd.read_csv("HybridRecommender-221114-235254/datasets/rating.csv")

# movieId, film adı ve filmin tür bilgilerini içeren veri seti
movies.head()
#UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti
ratings.head()

# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.
movies_ratings_df = pd.merge(ratings, movies, on="movieId")
movies_ratings_df.shape


# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız.Toplam oy kullanılma sayısı 1000'un altında olan filmleri veri setinden çıkarınız.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.
# Her film için kaç farklı kullanıcı oy kullanmış?
movie_vote_counts = movies_ratings_df.groupby("title")["rating"].count()


# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz
# 1000'den fazla oy alan filmlerin isimlerini alıyoruz
rare_movies = movie_vote_counts[movie_vote_counts < 1000].index
popular_movies = movie_vote_counts[movie_vote_counts >= 1000].index
common_movies_df = movies_ratings_df[movies_ratings_df["title"].isin(popular_movies)]

# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.
user_movie_df = common_movies_df.pivot_table(index="userId", columns="title", values="rating")

# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım
def create_user_movie_df():
    import pandas as pd
    movies = pd.read_csv("HybridRecommender-221114-235254/datasets/movie.csv")
    ratings = pd.read_csv("HybridRecommender-221114-235254/datasets/rating.csv")
    df = movies.merge(ratings, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.
random_user_id = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.
random_user_df = user_movie_df[user_movie_df.index == random_user_id]

# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
len(movies_watched)
#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.
movies_watched_df = user_movie_df[movies_watched]

# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
# Ve yeni bir df oluşturuyoruz.
user_movie_count = movies_watched_df.notna().sum(axis=1).reset_index()
user_movie_count.columns = ["userId", "movie_count"]
#user_movie_count.sort_values(by="movie_count", ascending=False)

# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"].tolist()
len(users_same_movies)
#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.

similar_users_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.
corr_df = similar_users_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()


# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.
top_users = corr_df[(corr_df["user_id_1"] == random_user_id) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz
top_users_ratings = top_users.merge(ratings[["userId", "movieId", "rating"]], how='inner')


#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# Adım 2: Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturunuz.
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)
movies_to_be_recommend.head()
# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.
movies_to_be_recommend.merge(movies[["movieId", "title"]]).head()

#############################################
# Görev 6: Item-Based Recommendation
#############################################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
user = 108170

# Adım 1: movie,rating veri setlerini okutunuz.
movies_ratings_df.head()


# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
most_recent_film = (
    movies_ratings_df
    .query("userId == @user and rating == 5")
    .sort_values("timestamp", ascending=False)
    .head(1)
)

most_recent_movie_id = int(most_recent_film["movieId"].values[0]) if not most_recent_film.empty else None

# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.
selected_movie_title = movies.loc[movies["movieId"] == most_recent_movie_id, "title"].values[0]

# user_movie_df'ten sadece bu film sütununu filtrele (kullanıcıların bu filme verdiği puanlar)
filtered_user_movie_df = user_movie_df[[selected_movie_title]]

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
movie_correlations = user_movie_df.corrwith(user_movie_df[selected_movie_title])


# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
# Korelasyonları DataFrame'e çevir
corr_df = pd.DataFrame(movie_correlations, columns=["correlation"])

top_similar_movies = movie_correlations.dropna().drop(labels=[selected_movie_title], errors='ignore').sort_values(ascending=False).head(5).to_frame(name='correlation')



"""
# NaN değerleri kaldır
corr_df.dropna(inplace=True)

# Korelasyonu en yüksek olanları sırala (kendisi hariç)
corr_df = corr_df.sort_values("correlation", ascending=False)

# En benzer 5 filmi al (kendisi dahilse kendisini filtrele)
top_similar_movies = corr_df[corr_df.index != selected_movie_title].head(5)

top_similar_movies
"""




