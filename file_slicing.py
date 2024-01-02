import os

# GitHub's file size limit is 100 MB for individual files
# but the Glove.6B dataset used to pre-train the Word2Vec model is around 1GB (10x capacity)

path = os.getcwd()
file_path = path + f'/Datasets/Input/glove.6B/'
output_path = path + f'/GPU_Server_Run/Input_Data/glove.6B/'

if os.path.exists(output_path) and not any(file.startswith('glove.6B.300d') for file in os.listdir(output_path)): # process must be repeate just once
    # chunk_size = 50000000  # 50MB chunk size to stay well below GitHub cap size
    n_chunks = 15
    file_number = 1

    with open(file_path + 'glove.6B.300d.txt', 'rb') as file: # read Binary Mode's better for large files
        content = file.readlines()
        n_rows = len(content)
        rows_per_chunk = n_rows // n_chunks # floor division
        for i in range(n_chunks):
            lower_bound = rows_per_chunk * i
            if i == (n_chunks - 1): # Catch the remainder
                upper_bound = n_rows
            else:
                upper_bound = rows_per_chunk * (i + 1)

            chunk = content[lower_bound : upper_bound]

            output_file_path = os.path.join(output_path, f'glove.6B.300d_n{file_number}.txt')
            with open(output_file_path, 'wb') as output_file: # write Binary Mode's better for large files
                output_file.writelines(chunk)

            file_number += 1
else:
    print(f'The process has been already run! Please check folder path {output_path}')
