from ultralytics import YOLO
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from PIL import Image
from django.db import connection
from rest_framework import status, permissions
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import Counter

@api_view(['POST'])
@permission_classes([permissions.AllowAny])
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
            'like': []
        }
        dish_data = ""
        for r in result:
            boxes = r.boxes.numpy()
            data = r.names
            for box in boxes:
                a = int(box.cls[0])
                dish_data = data[a]

        restaurants = []
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM restaurant")
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            for row in rows:
                data_col = dict(zip(columns, row))
                restaurants.append(data_col)

        for restaurant in restaurants:
            dishes = []
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM dish WHERE restaurant_id=%i", (restaurant['id'],))
                rows = cursor.fetchall()
                columns = [col[0] for col in cursor.description]
                for row in rows:
                    data_col = dict(zip(columns, row))
                    dish.append(data_col)

            for dish in dishes:
                dataOrder['product_name'].append(dish.title)
                dataOrder['did'].append(dish.did)
                dataOrder['is_featured'].append(dish.featured)
                dataOrder['like'].append(restaurant.like)

        results = search_restaurants(dataOrder, dish_data)
        sorted_results = sorted(results, key=lambda x: (x['is_featured'], x['like']), reverse=True)

        restaurant_results = []
        for result in sorted_results:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM dish WHERE id=%i", (result['did'],))
                rows = cursor.fetchone();
                if row is not None:
                    columns = [col[0] for col in cursor.description]
                    data_col = dict(zip(columns, row))
                    dish = data_col
            restaurant_results.append(dish)
                    
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
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_data)
    cosine_similarities = linear_kernel(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    similarities_with_names = list(zip(restaurants['product_name'], restaurants['did'], \
                                       restaurants['like'], restaurants['is_featured'], cosine_similarities))
    result = []
    for name, pid, like, is_featured, similarity in similarities_with_names:
        if similarity > 0.5:
            result.append({'product_name': name, 'did': pid, 'like': like, 'is_featured': is_featured})
    return result


@api_view(['GET'])
@permission_classes([permissions.AllowAny])
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
        cursor.execute("SELECT * FROM orders WHERE account_id=%i", (uid,))
        rows = cursor.fetchall()
        colunms = [col[0]for col in cursor.description]
        for row in rows:
            data = dict(zip(colunms, row))
            orders.append(data)

    for order in orders:
        orderItems = []
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM order_item WHERE orders_id=%i", (order['id'],))
            rows = cursor.fetchall()
            colunms = [col[0]for col in cursor.description]
            for row in rows:
                data = dict(zip(colunms, row))
                orderItems.append(data)
        for item in orderItems:
            dataOrder['product_name'].append(item.dish.title)
            dataOrder['order_date'].append(order.order_date.strftime("%Y-%m-%d"))
    
    dishes = []
    with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM dish WHERE restaurant_id=%i", (rid,))
            rows = cursor.fetchall()
            colunms = [col[0]for col in cursor.description]
            for row in rows:
                data = dict(zip(colunms, row))
                dishes.append(data)

    for dish in dishes:
        dataRestaurant['dataAll'].append(dish.title)
        dataRestaurant['did'].append(dish.did)
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