import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# Azure OpenAIのAPIに直接つなぐためのキー
api_key = st.text_input('apikey')
endpoint =  st.text_input('endpount')

st.title('LLMによるタスク処理')

llm = AzureChatOpenAI(
    deployment_name='gpt-35-turbo-0613', # モデル名
    openai_api_version='2023-05-15',
    azure_endpoint=endpoint, # クラウドにモデルを置いている場所
    openai_api_key=api_key, # アクセスするためのカギ
    temperature=1
)

user_input = st.text_area('プロンプトの入力')

# タスクの選択
template_option = st.selectbox('タスクの選択', ('要約タスク', 'ヒヤリハット報告要約'))

# テンプレートの辞書
templates = {
    '要約タスク' : '''以下の文章を要約してください。
                     {input}
                 ''',
    'ヒヤリハット報告要約' : '''以下の文章はヒヤリハット報告です。

{input}

上記を基に、必ず以下の形式で出力してください。
情報が含まれない場合、「不明」と出力してください。

・いつ
（ここに出力）

・どこで
（ここに出力）

・誰が
（ここに出力）

・なにをしていた
（ここに出力）

・なにが起きた
（ここに出力）

・原因
（ここに出力）
'''
}

st.write(templates[template_option])

prompt_template = PromptTemplate(
    input_variables=['input'],
    template=templates[template_option] # ユーザーが選んだテンプレート
)

output_button = st.button('GPTの出力')

if output_button:
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(user_input) # LLMから出力

    st.write(response)