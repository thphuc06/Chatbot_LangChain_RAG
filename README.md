# Chatbot_LangChain_RAG
Tuyệt vời\! Dưới đây là toàn bộ quy trình **FULL SETUP** để khởi động lại môi trường RAG của bạn trên máy chủ GPU mới, định dạng Markdown để bạn có thể lưu vào file `README.md` hoặc `setup.sh`.

-----

# 🚀 Quy Trình Khởi Động Lại Hệ Thống RAG (GPU Cloud)

Đây là các bước bạn cần thực hiện trên một máy chủ GPU mới, sau khi đã xóa máy chủ cũ.

## 1\. ⚙️ Cài đặt Công cụ Nền tảng (Ubuntu System)

Bạn cần cài đặt các công cụ cơ bản nhất như `git` và `python3` để bắt đầu.

```bash
# 1. Cập nhật danh sách gói (APT)
apt update

# 2. Cài đặt Python3, công cụ tạo venv, và Git
apt install -y python3 python3-venv git
```

## 2\. 📁 Lấy Code và Thiết lập Môi trường ảo

Bạn cần tải code về từ GitHub và tạo môi trường ảo (`venv`) để cô lập các thư viện.

```bash
# Thay thế URL bằng link repo của bạn
git clone https://github.com/thphuc06/Chatbot_LangChain_RAG.git

# Di chuyển vào thư mục dự án
cd Chatbot_LangChain_RAG

# 3. Tạo môi trường ảo (venv)
python3 -m venv venv

# 4. KÍCH HOẠT môi trường ảo (Bắt buộc phải làm mỗi lần SSH)
source venv/bin/activate
```

## 3\. 🔑 Xác thực Hugging Face (Quan trọng cho Llama-3)

Bạn cần Token để tải mô hình `Llama-3-8B-Instruct`.

```bash
# 1. Cài đặt công cụ CLI của Hugging Face
pip install huggingface-cli

# 2. Đăng nhập (Dán Access Token của bạn khi được yêu cầu)
huggingface-cli login
```

## 4\. 📦 Cài đặt Thư viện Python (LLM & RAG)

Bây giờ, bạn cài đặt tất cả các thư viện cần thiết. Lệnh này sẽ **tải các file mô hình** về máy chủ.

```bash
# Cài đặt tất cả thư viện (bao gồm cả các gói cần thiết cho quantization)
pip install -I --upgrade --force-reinstall pandas torch transformers accelerate bitsandbytes sentence-transformers langchain langchain-community chromadb
```

## 5\. 🏃 Hoàn thành và Chạy

Sau khi cài đặt xong, bạn có thể chạy code RAG của mình.

1.  **Mở VS Code:** Kết nối Remote-SSH lại, mở thư mục `Chatbot_LangChain_RAG`.
2.  **Chọn Kernel:** Trong Notebook, chọn Kernel là môi trường **`(venv)`** của bạn.
3.  **Chạy Code:** Chạy các cell code từ đầu (đặc biệt là cell tải mô hình Llama-3-8B).

-----

### ⚠️ Lưu Ý Khi Quay Lại Làm Việc

Mỗi lần bạn ngắt kết nối và quay lại SSH, bạn **bắt buộc** phải chạy 2 lệnh này:

```bash
# 1. Di chuyển vào thư mục code
cd Chatbot_LangChain_RAG

# 2. KÍCH HOẠT MÔI TRƯỜNG ẢO
source venv/bin/activate
```
