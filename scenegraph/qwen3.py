from openai import OpenAI
import json
import requests
import base64
import re
import time
from vl_prompt.p_manager import extract_objects, \
     get_discover_prompt,get_relationship_prompt,\
    judge_Correspondence_prompt,judge_OneObject_prompt,discover_OneObject_prompt,get_room_prompt,query_node_prompt_txt,target_node_prompt,query_state_transition


class LLM():
    def __init__(self):
        self.history = []
        self.model_name = "default"
        self.client = OpenAI(
        # api_key="", 
        # base_url="https://llmapi.blsc.cn/v1/"
        base_url = "http://192.168.100.24:8050/v1",api_key="EMPTY"
    )
    

    def get_room(self, img_name):
        from vl_prompt.prompt_cog.roomJudge import \
        SYSTEM_PROMPT, USER
        with open(img_name, "rb") as image_file:
            img = base64.b64encode(image_file.read()).decode('utf-8')

        completion = self.client.chat.completions.create(
            model = self.model_name, 
            messages= [{
                "role": "system", "content": SYSTEM_PROMPT
                }, {
                "role": "user", "content": [
                    {"type": "text", "text": USER}, 
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                ]
            }])
        reply = completion.choices[0].message.content
        return reply
    def query_target_obj(self,img_name,target_name):
        user=f"Is {target_name} in this image ?\n"
        from vl_prompt.prompt_gpt.target_object_makesure import \
            SYSTEM_PROMPT
        with open(img_name, "rb") as image_file:
            img = base64.b64encode(image_file.read()).decode('utf-8')
        completion = self.client.chat.completions.create(
            model = self.model_name, 
            messages= [{
                "role": "system", "content": SYSTEM_PROMPT
                }, {
                "role": "user", "content": [
                    {"type": "text", "text": user}, 
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                ]
            }])
        reply = completion.choices[0].message.content
        return reply

    def query_state_transition(self,user):
        from vl_prompt.prompt_gpt.state_transition import SYSTEM_PROMPT
        completion = self.client.chat.completions.create(
            model="default",
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user}
            ],
        )
        reply = completion.choices[0].message.content
        reply = json.loads(reply[reply.find("{"):reply.find("}")+1])
        return reply['Transition to state'],reply['Relative'] if 'Relative' in reply.keys() else None

    def query_node_txt(self, user):
        from vl_prompt.prompt_gpt.queryNodetxt import SYSTEM_PROMPT
        completion = self.client.chat.completions.create(
            model="default",
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user}
            ],
        )
        reply = completion.choices[0].message.content
        reply = json.loads(reply[reply.find("{"):reply.find("}")+1])
        if isinstance(reply['result'],int) :
            result = reply['result']
        else :
            result =  int(re.findall(r"\d+", reply['result'])[0])
        return reply,result
    

    def get_relationship_prompt(self,obj1,obj2,image):
        from vl_prompt.prompt_gpt.relation import \
            SYSTEM_PROMPT, USER1,USER2
        question = f"""{USER1}" "{obj1}" "{USER2}\t{obj2}"""
        with open(image, "rb") as image_file:
            img = base64.b64encode(image_file.read()).decode('utf-8')
        user_content =[{"type": "text", "text": question},{ "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"} }]
        completion = self.client.chat.completions.create(
            model="default",
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_content}
            ],
        )
        reply = completion.choices[0].message.content
        # print("reply:",reply)
        return reply
   