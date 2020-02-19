import json
import sys
from comp_detector4 import compliance_score
from mongo_test import mongodb_insert
from mongo_test import mongodb_get
from flask import Flask
from flask_cors import CORS
from flask import Flask, request
import pprint
import speechToText



app = Flask(__name__)

cors = CORS(app, resources={r"/getHistory": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/query": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/updateChatDetails": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


def Json_file_read():


    with open ("Watson1.json") as json_file:
        json_string = json_file.read()
        # print(json_string)

        data = json.loads(json_string)

        # print(data)
        text_data = data['results']
        # print(text_data1)
        
    # Results is an array, the [0] element of results is 'alternatives' which is also an array, the [0] element of alternatives is timestamps
    # which is also an array

        timestamps = data['results'][0]['alternatives'][0]['timestamps']
        transcript = data['results'][0]['alternatives'][0]['transcript']
        # print(transcript)

        speakers=data['speaker_labels']
        # print(speakers)


        ut = ""
        currentSpeaker = 0
        transcript_arr =[]
        temp_arr = []


        for q in timestamps:
            words = q[0]
            time1 = q[1]
            time2 = q[2]

            # print('time 1: ' + str(time1) + ' time 2: ' + str(time2))

            for p in speakers:
                spkSpeaker= p['speaker']
                spkstart= p['from']
                spkstop= p['to']

                if spkstart == time1:
            
                    if spkSpeaker == currentSpeaker:
                        ut +=  " " + words
                        # transcript_arr.append(str(spkSpeaker) + ": " + ut) #append value to the array
                        # print(currentSpeaker)

                    else:
                        # currentSpeaker = 0 #Toggle speaker
                        transcript_arr.append(str(spkSpeaker) + ": " + ut) #append value to the array
                        # temp_arr.append(json.dumps({spkSpeaker : ut}))
                        ut ='' #reset the 
                        if currentSpeaker == 1:
                            currentSpeaker = 0
                        else:
                            currentSpeaker = 1
                        ut == " " + words

    # pprint.pprint(transcript_arr)


    odd_i = []
    odd_i = transcript_arr[::2]
    non_compliant_sentences = []
    compliance_score_chat = 0.0
 
    isCompliant = "yes"
    compliance_check_sentences_count = 0
    for i in odd_i:
        Agent_sentence = (str(i)[2:])
        flag, f1, final_chunk = compliance_score(Agent_sentence)

        if flag == "No":
            non_compliant_sentences.append(Agent_sentence.strip())
            isCompliant = "no"
            compliance_score_chat = compliance_score_chat + float(f1)
            compliance_check_sentences_count+=1

        elif flag == "Yes":
            compliance_score_chat = compliance_score_chat + float(f1)
            compliance_check_sentences_count+=1

    confidence_score = int(compliance_score_chat)/compliance_check_sentences_count

    json_output = json.dumps({"transcript" : transcript_arr, "confidenceScore": confidence_score, "isCompliant": isCompliant , "compliance_check_chunk": non_compliant_sentences})
    # print(json_output)
    # mongodb_insert(json_output)
    return json_output



        
        

    # cmp_det = ComplianceDetector()

    # print("transcript: " + str(transcript_arr))
    # print("temparr:"+str(temp_arr))
    # flag,prob = cmp_det.predict(transcript_arr[len(transcript_arr) - 1])
    # flag,prob = compliance_score(transcript_arr[len(transcript_arr) - 1])

    # trans="thank you for calling customer services my name is a job offer how may I help you today hi subject on my name is Sean I've made a hotel booking a couple of months ago and I'm not feeling well so I'm looking forward to seeing if you can transfer from %HESITATION I'm so sorry to hear about it Sean I'll be glad to assist you on your reservation what I see yeah but that you have made a hotel reservation in doubletree by Hilton from sixteen to eighteen th of October %HESITATION I have checked the details and I see that it is inside the banality so what I need to do for you is I need to speak to the hotel to get a fever on the cancellation so may I place your call on hold while I quickly called the hotel and check with the allowing us to cancel this declaration without any charges yes please do that I advise you to attract thank you so much out of the line drops may I call you back the same number on which you're talking right now please do "
    # trans2="Thank you for calling customers so this is my name is the job the home may help you today hi Jonathan my name is Sean I've made a hotel booking a couple of months ago it's a non refundable booking but fallen ill and I'm looking to see if you can help me with canceling it shows on I'll be glad to assist you on this and I'm so sorry to hear about it let me just go ahead and get the details up on my screen hill so what I see that you have booked a hotel reservation in doubletree by Hilton from sixteen to eighteen of October I'm sure are checking up on the details I see that it is inside the penalty so I have to speak to the vendor to get a fever for the cancellation so give me like two minutes I'll quickly go to vendor take of either for the refund is that'll give you thank you so much for help I appreciate it"
    
    # flag, f1, final_chunk = compliance_score(Agent_sentence)

    # print(flag)
    # formatted_prob=(round(f1, 2)) 
    # print(formatted_prob)
    # print(final_chunk)


# Json_file_read()


@app.route('/getHistory')
def helloIndex():

    return mongodb_get()


@app.route("/Welcome/<name>") # at the endpoint /<name>
def Welcome_name(name): # call method Welcome_name
  return "Welcome" + name + "!" # which returns "Welcome + name + !


@app.route('/query')
def query_example():
    file_name= request.args.get('name') #if key doesn't exist, returns None
    # return '''<h1>The name value is: {}</h1>'''.format(name1)
    speechToText.fileName_transcript(str(file_name))
    fl_response = Json_file_read()
    return fl_response

@app.route("/updateChatDetails", methods=['POST']) # at the endpoint /<name>
def insertToDB(): # call method Welcome_name
    # if not request.json:
    #     abort(400)
    # print(request.json)
    # jsonObj = json.loads(request.json)
    # print(jsonObj)
    # chatDetails = {
    #     'transcript': jsonObj['transcript'],
    #     'confidenceScore': request.json['confidenceScore'],
    #     "isCompliant": request.json['isCompliant'],
    #     "compliance_check_chunk" : request.json['compliance_check_chunk'],
    #     "recID": request.json['recID']
    # }
    # print("HERE")
    # print(chatDetails)
    return mongodb_insert(request.json)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port= 1025)


