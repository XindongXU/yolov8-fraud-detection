from datetime import datetime
import logging
from paho.mqtt import client as mqtt_client

# variable globale, un flag pour vérifier si un code a été reçu
code_recu_flag = False

def connect_mqtt(broker, port, client_id):
    # fonction pour gérer la connexion au broker MQTT
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker :", broker)
        else:
            print("Failed to connect, return code %d\n", rc)

    # création d'une instance de client MQTT
    client = mqtt_client.Client(client_id)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def publish(client, topic, msg):
    # fonction pour publier un message sur un topic MQTT
    result = client.publish(topic, msg)
    if result[0] == 0:
        print(f"Send `{msg}` to topic `{topic}`")

# def subscribe(client: mqtt_client, topic):

#     global code_recu_flag
    
#     def on_message(client, userdata, msg):
    
#         global code_recu_flag

#         print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic at `{datetime.now()}`")
#         if "Accept True" in msg.payload.decode():
#             code_recu_flag = True
#             print(code_recu_flag)

#     client.subscribe(topic)
#     client.on_message = on_message

def subscribe(client: mqtt_client, topic, topic_pin, topic_pon, version):
    # fonction pour souscrire à un ou 
    # plusieurs topics MQTT et gérer les messages reçus
    global code_recu_flag
    
    def on_message(client, userdata, msg, topic = topic, topic_pin = topic_pin, topic_pon = topic_pon):
        global code_recu_flag
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic at `{datetime.now()}`")
        
        if msg.topic == topic_pin:
             # si le message est reçu sur le topic 'topic_pin'
             # publier un pong sur 'topic_pon'
            pong = f'pong-by IA version `{version}`-`{datetime.now()}`'
            result = client.publish(topic_pon, pong)
            if result[0] == 0:
                print(f"Send `{pong}` to topic `{topic_pon}`")
        if msg.topic == topic:
            # si le message est reçu sur le topic principal et contient "Accept True"
            # mettre à jour le flag
            if "Accept True" in msg.payload.decode():
                code_recu_flag = True
                # print(code_recu_flag)
    
    client.subscribe([(topic, 0), (topic_pin, 0)])
    client.on_message = on_message


def get_code_recu_flag(change = False):
    '''
        Créer une fonction qui renvoie la valeur de code_recu_flag 
        puis appeler cette fonction dans d'autres programmes 
        pour obtenir la dernière valeur de code_recu_flag
    '''
    
    global code_recu_flag
    # fait appel à la variable globale code_recu_flag définie dans ce script
    
    if not change:
        # si l'argument 'change' est False (qui est la valeur par défaut)
        # retournez simplement la valeur actuelle de code_recu_flag
        return code_recu_flag
    else:
        # réinitialisez la valeur de code_recu_flag à False
        code_recu_flag = False