from ultralytics import YOLO
from rest_framework.response import Response
from rest_framework.decorators import api_view
from PIL import Image
from django.db import connection
from rest_framework import status
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import Counter

@api_view(['POST'])
def search_restaurant_image(request):
    uploaded_image = request.FILES.get('image')
    image = Image.open(uploaded_image)
    try:
        model = YOLO("best.pt")
        result = model(image, stream=True)
        dataOrder = {
            'product_name' : [],
            'did': [],
            'is_featured': [],
            'likes': []
        }
        dish_data = ""
        for r in result:
            boxes = r.boxes.numpy()
            data = r.names
            for box in boxes:
                a = int(box.cls[0])
                dish_data = data[a]
        # print(dish_data)

        restaurants = []
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM restaurant")
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            for row in rows:
                data_col = dict(zip(columns, row))
                restaurants.append(data_col)

        # print(restaurants)

        for restaurant in restaurants:
            dishes = []
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM dish WHERE restaurant_id=%s", (restaurant['id'],))
                rows = cursor.fetchall()
                columns = [col[0] for col in cursor.description]
                for row in rows:
                    data_col = dict(zip(columns, row))
                    dishes.append(data_col)
            # print( dishes)

            for dish in dishes:
                dataOrder['product_name'].append(dish['title'])
                dataOrder['did'].append(dish['id'])
                dataOrder['is_featured'].append(dish['featured'])
                dataOrder['likes'].append(restaurant['likes'])

        # print(dataOrder, " ok")
        results = search_restaurants(dataOrder, dish_data)
        sorted_results = sorted(results, key=lambda x: (x['is_featured'], x['likes']), reverse=True)
        print(sorted_results)

        restaurant_results = []
        dish_ids = [result['did'] for result in sorted_results]
        dish_restaurant_ids = [(dish_id, ) for dish_id in dish_ids]
        with connection.cursor() as cursor:
            cursor.executemany("SELECT id, restaurant_id FROM dish WHERE id IN (%s)", dish_restaurant_ids)
            dish_rows = cursor.fetchall()
            restaurant_ids = [row[1] for row in dish_rows]
            restaurant_id_tuples = [(restaurant_id, ) for restaurant_id in restaurant_ids]
            cursor.executemany("SELECT * FROM restaurant WHERE id IN (%s)", restaurant_id_tuples)
            restaurant_rows = cursor.fetchall()
            for restaurant_row in restaurant_rows:
                restaurant_dict = dict(zip([col[0] for col in cursor.description], restaurant_row))
                restaurant_results.append(restaurant_dict)
                    
        return Response({
            'success': True,
            'message': "Search successfully.",
            'data': {
                'result': dish_data,
                'restaurant': restaurant_results
            }
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({
            'success': False,
            'message': str(e)
        }, status=status.HTTP_200_OK)
    
def search_restaurants(restaurants, query):
    all_data = restaurants['product_name'] + [query]
    # Tạo ma trận TF-IDF cho các sản phẩm loại bỏ từ dừng
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_data)
    # Sử dụng linear_kernel để tính toán độ tương đồng giữa "Mì Quảng" và tất cả các sản phẩm khác
    cosine_similarities = linear_kernel(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    similarities_with_names = list(zip(restaurants['product_name'], restaurants['did'], \
                                       restaurants['likes'], restaurants['is_featured'], cosine_similarities))
    result = []
    for name, pid, likes, is_featured, similarity in similarities_with_names:
        if similarity > 0.5:
            result.append({'product_name': name, 'did': pid, 'likes': likes, 'is_featured': is_featured})
    return result


@api_view(['GET'])
def suggest_food(request, *args, **kwargs):
    uid = kwargs.get('uid')
    rid = kwargs.get('rid')
    dataRestaurant = {
        'did': [],
        'dataAll': []
    }
    dataOrder = {
        'product_name': [],
        'order_date': []
    }
    
    orders = []
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM orders WHERE account_id=%s", (uid,))
        rows = cursor.fetchall()
        colunms = [col[0]for col in cursor.description]
        for row in rows:
            data = dict(zip(colunms, row))
            orders.append(data)

    # print(orders)

    for order in orders:
        orderItems = []
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM order_item WHERE order_id=%s", (order['id'],))
            rows = cursor.fetchall()
            colunms = [col[0]for col in cursor.description]
            for row in rows:
                data = dict(zip(colunms, row))
                orderItems.append(data)
            
        for item in orderItems:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM dish WHERE id=%s", (item['dish_id'],))
                rows = cursor.fetchall()
                colunms = [col[0]for col in cursor.description]
                for row in rows:
                    data = dict(zip(colunms, row))
                    dataOrder['product_name'].append(data['title'])
            dataOrder['order_date'].append(order['date'].strftime("%Y-%m-%d"))
    
    dishes = []
    with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM dish WHERE restaurant_id=%s", (rid,))
            rows = cursor.fetchall()
            colunms = [col[0]for col in cursor.description]
            for row in rows:
                data = dict(zip(colunms, row))
                dishes.append(data)

    for dish in dishes:
        dataRestaurant['dataAll'].append(dish['title'])
        dataRestaurant['did'].append(dish['id'])
    # print(dataRestaurant)
    data = find_similarities(dataRestaurant, dataOrder)

    return Response({
            'success': True,
            'message': 'Suggest Successful.',
            'data': data
        }, status=status.HTTP_200_OK)

def find_similarities(dataRestaurant, dataOrder):
    data = pd.DataFrame(dataOrder)
    data['order_date'] = pd.to_datetime(data['order_date'])
    data = data.sort_values(by='order_date', ascending=False)
    top_products = [item[0] for item in Counter(data['product_name']).most_common(5)]
    result = []
    for product in top_products:
        all_data = dataRestaurant['dataAll'] + [product]
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_data)
        cosine_similarities = linear_kernel(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        similarities_with_names = list(zip(dataRestaurant['dataAll'], dataRestaurant['did'], cosine_similarities))
        for dataAll, did, similarity in similarities_with_names:
            if similarity >= 0.4:
                result.append({'did': did, 'title': dataAll})
    return result