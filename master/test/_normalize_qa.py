# import json
# import uuid

# origin_qa_data_path = 'dataset/QA Data/MedMCQA/hard_questions.jsonl'
# target_qa_data_path = 'dataset/QA Data/MedMCQA/translated_hard_questions.jsonl'

# def transform_id(origin_id):
#     # Add 'T' prefix and remove last character
#     return ' T' + origin_id[:-1]

# def update_answers():
#     # Read origin data
#     with open(origin_qa_data_path, 'r', encoding='utf-8') as f:
#         origin_data = [json.loads(line) for line in f]

#     # Read target data
#     with open(target_qa_data_path, 'r', encoding='utf-8') as f:
#         target_data = [json.loads(line) for line in f]

#     c = []
#     for item in origin_data:
#         for target_item in target_data:
#             if transform_id(item['id']) == target_item['uuid']:
#                 if item['cop'] == 0:
#                     target_item['answer'] = 'A'
#                 elif item['cop'] == 1:
#                     target_item['answer'] = 'B'
#                 elif item['cop'] == 2:
#                     target_item['answer'] = 'C'
#                 elif item['cop'] == 3:
#                     target_item['answer'] = 'D'
#                 c.extend([target_item['uuid']])
#     # print(c)
#     for item in target_data:
#         if item['uuid'] not in c:
#             print(item['uuid'])
#     # Write updated target data back to file
#     with open(target_qa_data_path, 'w', encoding='utf-8') as f:
#         for item in target_data:
#             f.write(json.dumps(item, ensure_ascii=False) + '\n')

# # Call the function to update answers
# update_answers()