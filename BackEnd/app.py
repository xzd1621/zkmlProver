import ezkl
from flask import Flask, request, redirect, url_for, render_template_string, send_file
import os

from flask_cors import CORS
from werkzeug.utils import secure_filename
import zipfile

import Agg_Testing
from Agg_Testing import gen_verifier  # 这里假设gen_verifier函数在some_module模块中
import shutil

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "http://localhost:3000"}})
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'zip', 'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def extract_zip(input_zip_path, output_dir_path):
    with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir_path)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    pk_path = os.path.join('test.pk')
    proof_path = os.path.join('proof.json')
    vk_path = os.path.join('test.vk')
    settings_path = os.path.join('settings.json')
    srs_path = os.path.join('kzg.srs')
    if request.method == 'POST':
        zip_file = request.files['photozip']
        target_file = request.files['targetphoto']

        if zip_file and allowed_file(zip_file.filename) and target_file and allowed_file(target_file.filename):
            zip_filename = secure_filename(zip_file.filename)
            target_filename = secure_filename(target_file.filename)

            zip_filepath = os.path.join(app.config['UPLOAD_FOLDER'], zip_filename)
            target_filepath = os.path.join(app.config['UPLOAD_FOLDER'], target_filename)

            zip_file.save(zip_filepath)
            target_file.save(target_filepath)

            output_dir_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted')
            if os.path.exists(output_dir_path):
                shutil.rmtree(output_dir_path)  # 删除之前的解压文件夹
            os.makedirs(output_dir_path)
            extract_zip(zip_filepath, output_dir_path)

            clean_model = Agg_Testing.clean_model
            target_filepath = Agg_Testing.target
            gen_verifier(clean_model, target_filepath, True)
            sol_code_path = os.path.join('Verifier.sol')
            abi_path = os.path.join('Verifier.abi')

            res = ezkl.create_evm_verifier(
                vk_path,
                srs_path,
                settings_path,
                sol_code_path,
                abi_path
            )

            assert res == True
            assert os.path.isfile(sol_code_path)

            # res = ezkl.verify(
            #     proof_path,
            #     settings_path,
            #     vk_path,
            #     srs_path,
            # )
            #
            # assert res == True
            # print("verified res:", res)

            # 创建一个ZIP文件来保存所有的文件
            zip_filename = 'Verifier.zip'
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                zipf.write(proof_path, arcname='proof.txt')  # arcname是在zip文件中的名字
                zipf.write(settings_path, arcname='settings.json')
                zipf.write(vk_path, arcname='test.vk')
                zipf.write(srs_path, arcname='kzg.srs')
                zipf.write(sol_code_path, arcname='Verifier.sol')


            # 返回ZIP文件给前端
            return send_file(zip_filename, as_attachment=True, download_name='Verifier.zip')
            # return send_file(sol_code_path, as_attachment=True, download_name='Verifier.sol')


    return '''
    <!doctype html>
    <title>Upload a zip of photos and a target photo</title>
    <h1>Upload a zip of photos and a target photo</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=photozip>
      <input type=file name=targetphoto>
      <input type=submit value=Upload>
    </form>
    '''


if __name__ == '__main__':
    app.run(debug=True)
