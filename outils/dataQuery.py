from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

def SendQuery(es_username, es_password, id_machine, startTime, stopTime, es_index, size):
    # Connection to Elasticsearch
    es = Elasticsearch(hosts = "https://wapp-gbg-elasticsearch-dev.azurewebsites.net:443/elastic/",
                       basic_auth = (es_username, es_password))
    query_body = {
      "query": {
        "bool": {
          "must": [
            {
              "match": {
                "machineSerialNumber": id_machine
              }
            },
            {
              "range": {
                "createdAt": {
                  "gte": startTime,
                  "lte": stopTime
                }
              }
            }
          ]
        }
      },
        "size": size  # You can adjust the size as needed
    }

    response = es.search(index=es_index, body=query_body)
    
    if es_index == "stream-reporting-sessions":
        hits = response['hits']['hits']
        data = [hit['_source'] for hit in hits]
        data_ihm = pd.DataFrame(data)
        column_list = ['machineSerialNumber', 'sessionStart', 'sessionStop', 'nbBottle']
        data_ihm = data_ihm[column_list]

        data_ihm['sessionStart'] = pd.to_datetime(data_ihm['sessionStart'])
        data_ihm['sessionStop'] = pd.to_datetime(data_ihm['sessionStop'])
        data_ihm['sessionStart-1s'] = data_ihm['sessionStart'] - pd.Timedelta(seconds=1)
#         data_ihm['sessionStop'] = data_ihm['sessionStop'] + pd.Timedelta(hours=2)# + pd.Timedelta(seconds=6)
        data_ihm['sessionStart'] = data_ihm['sessionStart'].dt.tz_localize(None)
        data_ihm['sessionStart-1s'] = data_ihm['sessionStart-1s'].dt.tz_localize(None)
        data_ihm['sessionStop'] = data_ihm['sessionStop'].dt.tz_localize(None)

        data_ihm = data_ihm.sort_values(by='sessionStart-1s', ascending=True)
        column_list = ['machineSerialNumber', 'sessionStart-1s', 'sessionStop', 'nbBottle']
        data_ihm = data_ihm[column_list]

        data_ihm.reset_index(drop=True, inplace=True)
        data_ihm.index += 1
        data_ihm.index.name = 'session_id'
        return data_ihm
    
    elif es_index == "stream-reporting-packings":
        hits = response['hits']['hits']
        data = [hit['_source'] for hit in hits]
        data_packing = pd.DataFrame(data)
        column_list = ['machineSerialNumber', 'createdAt', 'accept', 'barCode', 'brand', 'reason']
        data_packing = data_packing[column_list]

        data_packing.columns = ['machineSerialNumber', 'actionTime', 'isAccept', 'barCode', 'brand', 'reason']

        data_packing['actionTime'] = pd.to_datetime(data_packing['actionTime'])
        data_packing['actionTime'] = data_packing['actionTime'].dt.tz_localize(None)

        data_packing = data_packing.sort_values(by='actionTime', ascending=True)
        data_packing.reset_index(drop=True, inplace=True)
        data_packing.index += 1
        data_packing.index.name = 'packing_id'

        return data_packing