from pymongo import MongoClient
import json
import pprint

client = MongoClient()


client = MongoClient('localhost', 27017)
db = client['VCA']
posts = db.posts



def mongodb_insert(text):
    print(text)
    parsed_json=json.loads(json.dumps(text))
    post_data = {
        'transcript': parsed_json['transcript'],
        'confidenceScore': parsed_json['confidenceScore'],
        "isCompliant": parsed_json['isCompliant'],
        "compliance_check_chunk" : parsed_json['compliance_check_chunk'],
        "recID": parsed_json['recID']
    }
    result = posts.insert_one(post_data)
    return('One post: {0}'.format(result.inserted_id))




def mongodb_get():
    vca_table = posts.find({}, {'_id': 0})
    ut = []
    for items in vca_table:
        ut.append(items)

    print (json.dumps(ut))
    return json.dumps(ut)

    
    # return str(bills_post)
    # print(ut)
    # # print('One post: {0}'.format(result.title))
    # # return('One post: {0}'.format(result.))
    # print(ut)


# mongodb_get()


