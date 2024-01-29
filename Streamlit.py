import os
import moviepy.editor as mp
import speech_recognition as sr
import csv
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import base64

def konversi_video_ke_teks(file_path):
    try:
        # Dapatkan path lengkap ke file video
        path_video = os.path.join(os.path.dirname(__file__), file_path)

        # Ekstrak audio dari video
        video = mp.VideoFileClip(path_video)
        video.audio.write_audiofile("temp_audio.wav", codec='pcm_s16le')

        # Inisialisasi objek recognizer
        recognizer = sr.Recognizer()

        # Baca file audio
        with sr.AudioFile("temp_audio.wav") as file_audio:
            audio = recognizer.record(file_audio)

        # Pengenalan suara menggunakan Google Web Speech API
        hasil_teks = recognizer.recognize_google(audio, language="id-ID")
        return hasil_teks

    except FileNotFoundError:
        return f"File '{file_path}' tidak ditemukan di folder yang sama dengan skrip."

    except sr.UnknownValueError:
        return "Google Web Speech API tidak dapat mengenali audio."

    except sr.RequestError as e:
        return f"Error pada permintaan API: {e}"

    except Exception as e:
        return f"Terjadi kesalahan: {e}"

# Mendapatkan path lengkap ke file skrip
path_script = os.path.dirname(os.path.abspath(__file__))

# Fungsi untuk mengonversi skor sentimen menjadi label
def skor_ke_label(skor):
    if skor > 0.5:
        return 'Positif'
    else:
        return 'Negatif'

def download_file(content, file_name, file_type):
    base64_encoded = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/{file_type};base64,{base64_encoded}" download="{file_name}">Download {file_name}</a>'
    return href

def main():
    # Judul dan deskripsi halaman Streamlit
    st.title("Aplikasi Konversi Video ke Teks dan Analisis Sentimen")
    st.write("Aplikasi ini dapat melakukan konversi teks dari audio video, menganalisis sentimen teks, dan menampilkan WordCloud.")

    # Input atau upload file video menggunakan Streamlit
    file_path_video = st.file_uploader("Pilih atau upload file video (hanya dukungan format mp4)", type=["mp4"])

    # Cek jika file video sudah diinput atau diupload
    if file_path_video is not None:
        # Contoh penggunaan dengan video yang diinput atau diupload
        hasil_teks = konversi_video_ke_teks(file_path_video.name)

        # Menampilkan hasil konversi
        st.subheader("Hasil Konversi Teks:")
        st.write(hasil_teks)

        # Simpan hasil konversi ke dalam file teks di folder yang sama dengan skrip
        output_file_name = f"hasil_konversi_{os.path.basename(file_path_video.name)}.txt"
        output_file_path = os.path.join(path_script, output_file_name)
        with open(output_file_path, "w", encoding="utf-8") as file_teks:
            file_teks.write(hasil_teks)

        st.success(f"Hasil konversi telah disimpan di '{output_file_path}'.")
        st.markdown(download_file(hasil_teks, output_file_name, 'txt'), unsafe_allow_html=True)

        # Load model sentimen bahasa Indonesia
        sentimen_analisis = pipeline('sentiment-analysis', model='indobenchmark/indobert-base-p2')

        # Analisis sentimen teks
        hasil_sentimen = sentimen_analisis(hasil_teks)

        # Tampilkan hasil sentimen dalam bentuk label 'positif' atau 'negatif'
        st.subheader("Hasil Analisis Sentimen:")
        for hasil in hasil_sentimen:
            label = skor_ke_label(hasil['score'])
            st.write(f"Sentimen: {label}")

        # Dapatkan path lengkap ke file CSV
        path_csv = os.path.join(path_script, f'hasil_sentimen_{os.path.basename(file_path_video.name)}.csv')

        # Simpan hasil ke dalam file CSV
        with open(path_csv, 'w', newline='', encoding='utf-8') as file_csv:
            writer = csv.writer(file_csv)
            writer.writerow(['Hasil Konversi', 'Analisis Sentimen'])

            # Menulis setiap baris ke dalam file CSV
            for hasil in hasil_sentimen:
                label = skor_ke_label(hasil['score'])
                writer.writerow([hasil_teks, label])

        st.success(f"Hasil sentimen telah disimpan dalam file '{path_csv}'.")
        st.markdown(download_file('\n'.join([f"{hasil_teks},{skor_ke_label(hasil['score'])}" for hasil in hasil_sentimen]), os.path.basename(path_csv), 'csv'), unsafe_allow_html=True)

        # Dapatkan path lengkap ke file 'hasil_konversi.txt'
        file_path_txt = os.path.join(path_script, output_file_name)

        # Tambahkan try-except block untuk WordCloud
        try:
            with open(file_path_txt, 'r', encoding='utf-8') as file:
                teks = file.read()

            # Inisialisasi WordCloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(teks)

            # Simpan WordCloud ke file gambar di folder yang sama dengan skrip
            image_path = os.path.join(path_script, f'wordcloud_image_{os.path.basename(file_path_video.name)}.png')
            wordcloud.to_file(image_path)

            # Tampilkan WordCloud
            st.subheader("Hasil WordCloud:")
            st.image(wordcloud.to_array(), use_container_width=True)

            st.success(f"Hasil WordCloud telah disimpan di '{image_path}'.")
            st.markdown(download_file(image_path, os.path.basename(image_path), 'png'), unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning(f"File '{file_path_txt}' tidak ditemukan.")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# Panggil fungsi main()
if __name__ == "__main__":
    main()
