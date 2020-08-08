from qamodel.bert import QA
model = QA('model')


def get_answer(context:str, ques:str):
    answer= model.predict(context, ques)
    return answer['answer']

context = "Victoria has a written constitution enacted in 1975, but based on the 1855 colonial constitution, passed by the United Kingdom Parliament as the Victoria Constitution Act 1855, which establishes the Parliament as the state's law-making body for matters coming under state responsibility. The Victorian Constitution can be amended by the Parliament of Victoria, except for certain 'entrenched' provisions that require either an absolute majority in both houses, a three-fifths majority in both houses, or the approval of the Victorian people in a referendum, depending on the provision."
ques1 = "When did Victoria enact constitution"
ques2= "Who passed the constitution in 1855"
answer1 = get_answer(context, ques1)
answer2 = get_answer(context, ques2)

print(answer1)
print('\n')
print(answer2)


# In[ ]:




