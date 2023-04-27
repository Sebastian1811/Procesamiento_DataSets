import w2v


def get_cosine_similarity_recommendations(beca_name):
    recommendations = w2v.recommendations(beca_name) 
    print(recommendations)

get_cosine_similarity_recommendations("Becas SRE – Universidad Autónoma de Coahuila (UAdeC)")