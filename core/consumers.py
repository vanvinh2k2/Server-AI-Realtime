from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.db import connection
from core.models import *
import json
import asyncio

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.username = self.scope["url_route"]["kwargs"]["username"]
        if self.username is None:
            await self.close()
            return
        
        await self.channel_layer.group_add(
            self.username,
            self.channel_name
        )
        await self.accept()

    @database_sync_to_async
    def get_user_from_username(self, username):
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM account WHERE account.username = %s", (username,))
                row = cursor.fetchone()
                if row is not None:
                    columns = [col[0] for col in cursor.description]
                    user = dict(zip(columns, row))
                    return user
        except :
            return None
    
    @database_sync_to_async
    def create_message(self, send_id, recipient_id, body):
        try:
            with connection.cursor() as cursor:
                cursor.execute("INSERT INTO chat_message (send_id, recipient_id, body, date) \
                VALUES (%s, %s, %s, NOW())", (send_id, recipient_id, body))
            connection.commit()
            return True
        except: return False
        
    @database_sync_to_async
    def execute_sql_query(self, user_id, friend_id):
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM chat_message AS c \
                               WHERE (c.send_id = %s OR c.recipient_id = %s)\
                               AND (c.send_id = %s OR c.recipient_id = %s)\
                               ORDER BY c.date ASC",\
                               (user_id, user_id, friend_id, friend_id))
                columns = [col[0] for col in cursor.description]
                chatmessage = []
                for row in cursor.fetchall():
                    row_dict = dict(zip(columns, row))
                    row_dict['date'] = row_dict['date'].strftime('%Y-%m-%d %H:%M:%S')
                    chatmessage.append(row_dict)
                return chatmessage
        except Exception as e:
            print(f"Error fetching chat messages: {e}")
            return None
        
    async def disconnect(self):
        # Roi chat
        await self.channel_layer.group_discard(
            self.username, self.channel_name
        )

    async def receive(self, text_data):
        data = json.loads(text_data)
        data_source = data.get('source')
        print('receive', json.dumps(data, indent=2))
        if data_source == 'message-list':
            await self.receive_message_list(data)
        if data_source == 'message-user':
            await self.send_message_user(data)

    async def receive_message_list(self, data):
        username = data.get('friend')
        user = await self.get_user_from_username(self.username)
        friend = await self.get_user_from_username(username)
        print(user['id'], friend['id'])
        chatmessage = chatmessage = await self.execute_sql_query(user['id'], friend['id'])
        if chatmessage is not None:
            await self.send_group(self.username, 'message-list', chatmessage)
            await self.send_group(username, 'message-list', chatmessage)

    async def send_message_user(self, data):
        username = data.get('friend')
        message = data.get('message')
        user = await self.get_user_from_username(self.username)
        friend = await self.get_user_from_username(username)
        status = await self.create_message(user['id'], friend['id'], message)
        if(status): chatmessage = await self.execute_sql_query(user['id'], friend['id'])

        if chatmessage is not None:
            await self.send_group(self.username, 'message-list', chatmessage)
            await self.send_group(username, 'message-list', chatmessage)


    async def send_group(self, group, source, data):
        response = {
            'type': 'broadcast_group',
            'source': source,
            'data': data
        }
        await self.channel_layer.group_add(
            group, self.channel_name
        )
        await self.channel_layer.group_send(
            group, response
        )

    async def broadcast_group(self, event):
        message_data = event['data']
        asyncio.create_task(self.send(text_data=json.dumps(message_data)))