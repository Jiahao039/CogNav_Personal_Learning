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
        
        # [修改 1] 配置通义千问的模型名称 (推荐 qwen-plus 或 qwen-max)
        self.model_name = "qwen-plus"
        
        # [修改 2] 配置通义千问的 Base URL 和 API Key
        # 注意：请将下面的 sk-xxx 替换为你真实的 API Key
        self.api_key = "sk-57b4f55a64d04687801c239416eeef54" 
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
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
        # [修改 3] 将 "default" 改为 self.model_name，否则阿里云API会报错找不到模型
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user}
            ],
        )
        reply = completion.choices[0].message.content
        # 增加容错处理，防止返回内容没有大括号
        try:
            json_str = reply[reply.find("{"):reply.rfind("}")+1]
            reply_json = json.loads(json_str)
            return reply_json['Transition to state'], reply_json.get('Relative', None)
        except Exception as e:
            print(f"Error parsing JSON from LLM response: {reply}")
            # 返回一个默认的安全状态，防止程序崩溃
            return "Broad Search", None

    def query_node_txt(self, user):
        from vl_prompt.prompt_gpt.queryNodetxt import SYSTEM_PROMPT
        # [修改 4] 将 "default" 改为 self.model_name
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user}
            ],
        )
        reply = completion.choices[0].message.content
        
        try:
            json_str = reply[reply.find("{"):reply.rfind("}")+1]
            reply_json = json.loads(json_str)
            
            if isinstance(reply_json['result'], int):
                result = reply_json['result']
            else:
                # 尝试提取数字
                nums = re.findall(r"\d+", str(reply_json['result']))
                result = int(nums[0]) if nums else -1
            return reply_json, result
        except Exception as e:
            print(f"Error parsing JSON from LLM response: {reply}")
            return {"result": -1}, -1
    

    def get_relationship_prompt(self,obj1,obj2,image):
        from vl_prompt.prompt_gpt.relation import \
            SYSTEM_PROMPT, USER1,USER2
        question = f"""{USER1}" "{obj1}" "{USER2}\t{obj2}"""
        with open(image, "rb") as image_file:
            img = base64.b64encode(image_file.read()).decode('utf-8')
        user_content =[{"type": "text", "text": question},{ "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"} }]
        
        # [修改 5] 将 "default" 改为 self.model_name
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_content}
            ],
        )
        reply = completion.choices[0].message.content
        # print("reply:",reply)
        return reply