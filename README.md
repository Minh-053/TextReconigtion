**ĐỀ TÀI: NHẬN DIỆN KÝ TỰ QUANG HỌC**

Nhận diện ký tự quang học (OCR - Optical Character Recognition) là một bài toán kinh điển trong lĩnh vực Thị giác máy tính, cho phép máy tính đọc hiểu văn bản từ hình ảnh. Với sự phát triển của học sâu, các mô hình hiện đại như VietOCR đã được xây dựng và huấn luyện để nhận diện văn bản tiếng Việt với độ chính xác cao. Đồ án này chúng em tập trung nghiên cứu, triển khai và đánh giá hiệu quả của mô hình VietOCR trong việc nhận diện văn bản từ ảnh

**Mục tiêu:**

Chúng em đề ra các mục tiêu như sau: 
-  Hiểu rõ nguyên lý hoạt động của mô hình VietOCR.
-  Huấn luyện và đánh giá mô hình VietOCR trên tập dữ liệu chứa văn bản tiếng Việt.
-  Tiền xử lý ảnh để tăng độ chính xác nhận diện.
-  Triển khai hệ thống nhận diện ảnh đầu vào và xuất văn bản.
-  So sánh mô hình đã huấn luyện với mô hình pre-trained gốc.
  
**Cơ sở lý thuyết:**

1. Tìm hiểu về Việt OCR

VietOCR là một mô hình nhận diện văn bản (Optical Character Recognition - OCR) mã nguồn mở, được thiết kế và tối ưu hóa cho văn bản tiếng Việt. Nó được xây dựng dựa trên kiến trúc học sâu encoder-decoder, có khả năng nhận diện các chuỗi ký tự từ hình ảnh văn bản một cách chính xác.

a. Kiến trúc tổng quan

VietOCR gồm các thành phần chính sau:
Backbone (CNN):
- Mục tiêu là trích xuất đặc trưng không gian từ ảnh đầu vào.
- Các lựa chọn backbone có thể là vgg, resnet, hoặc efficientnet.
Encoder:
- Nếu sử dụng kiến trúc Transformer, encoder sẽ học mối   quan hệ không gian giữa các đặc trưng ảnh.
- Nếu dùng kiến trúc seq2seq, phần encoder đóng vai trò rút trích thông tin chuỗi từ ảnh.
Decoder:
- Dựa trên kết quả từ encoder, decoder sinh ra chuỗi ký tự đầu ra một cách tuần tự.
  
b. Các kiến trúc hỗ trợ

VietOCR hỗ trợ một số kiến trúc huấn luyện chính:
- vgg_transformer
- vgg_seq2seq
- resnet_transformer
- efficientnet_lite0_transformer
Trong đó:
- vgg_seq2seq là kiến trúc phổ biến nhất, đơn giản, dễ train, thích hợp cho ảnh văn bản thông thường. => Đó là lý do chúng em chọn mô hình huấn luyện này.
- transformer thích hợp hơn với văn bản phức tạp, dài, nhưng cần nhiều tài nguyên.
- 
c. Định dạng dữ liệu

- Dữ liệu huấn luyện gồm các cặp (ảnh, chuỗi văn bản).
- VietOCR hỗ trợ format .txt chứa đường dẫn ảnh và ground truth, ví dụ:
Tên ảnh: 17.jpg

 ![image](https://github.com/user-attachments/assets/3bde823f-59c7-4c29-bffb-fd31c828091e)
 
Thì nhãn (label) sẽ là :

 ![image](https://github.com/user-attachments/assets/ecce351e-d1f4-49f1-a94b-8d78d700ecd0)
 
Trong này:
-  “train/17.jpg ” là đường dẫn đến ảnh gốc.
- “PHƯỜNG ĐÔNG XUYÊN, THÀNH PHỐ LONG XUYÊN, TỈNH AN GIANG” là nhãn của ảnh.
  
2. Tìm hiểu về Tesseract OCR - Công cụ phát hiện vùng văn bản
3. 
Tesseract OCR là một trong những thư viện nhận diện ký tự quang học (OCR) mã nguồn mở mạnh mẽ và phổ biến nhất hiện nay, được phát triển ban đầu bởi HP và sau này do Google duy trì. Không chỉ nhận diện ký tự, Tesseract còn cung cấp khả năng phân tích cấu trúc bố cục văn bản trong ảnh, bao gồm:
- Xác định các dòng văn bản (lines)
- Cắt thành vùng từ (words)
- Trích xuất bounding box (tọa độ chữ, dòng hoặc đoạn)

Trong đề tài này, Tesseract OCR được sử dụng không phải để nhận diện nội dung, mà để tách ảnh gốc thành các dòng văn bản riêng biệt, sau đó:
1.	Mỗi dòng được cắt ra dựa trên bounding box.
2.	Đưa các dòng này qua mô hình VietOCR để nhận diện.

Ưu điểm:
- Giảm chiều dài ảnh đầu vào → giúp mô hình VietOCR dễ học hơn.
- Tách dòng giúp nhận diện chính xác hơn (tránh nhầm lẫn giữa các dòng).
- Tesseract hoạt động nhanh và không cần huấn luyện lại.

**Triền khai thực tiễn**

. Thu thập dữ liệu 

-  Dữ liệu gồm các ảnh chứa văn bản tiếng Việt, thu thập từ sách, tài liệu, chữ viết tay.
-  Dạng dữ liệu: cặp (ảnh, chuỗi text đúng).
1 số ví dụ :
Ảnh:

![image](https://github.com/user-attachments/assets/4979d92c-6b88-43c7-959a-2cbaa2762e4b)

Nhãn: “Bức” ““Xuống Đồng”” Của Trần Văn Cẩn Đã Được Trao Giải Nhất Và Được Hội Văn Hóa Cứu Quốc Mua”

Ảnh: 

 ![image](https://github.com/user-attachments/assets/1f44a571-de42-41c9-8081-ad2ce06ced31)

Nhãn: Thành công ấy, liệu có mấy người đạt được ?




Bộ dataset của nhóm chúng em thu thập được vào khoảng hơn 4000k ảnh 

Trong đó :

Thư mục train(Training set) bao gồm 80% - 3200 ảnh:

- **Mục đích**: Dùng để huấn luyện mô hình – tức là giúp mô hình học được mối quan hệ giữa ảnh và văn bản.
- **Vai trò**: Làm mô hình học được cách dự đoán đầu ra đúng (chuỗi ký tự) từ ảnh đầu vào.

Thư mục val(Validation) bao gồm 20% - 800 ảnh:
-  **Mục đích**: Dùng để đánh giá mô hình trong quá trình huấn luyện, tức là kiểm tra xem mô hình có đang học tốt hay bị overfit (học tủ).
- **Vai trò:**
+ Giúp theo dõi độ chính xác (accuracy) và độ lỗi (loss) sau mỗi vòng lặp (epoch).
+ Nếu mô hình có validation loss tăng trong khi training loss giảm → có thể đang bị overfitting.
  
  2. Bắt đầu train mô hình
  
Quá trình huấn luyện mô hình được thực hiện bằng cách sử dụng thư viện VietOCR, với kiến trúc vgg_seq2seq. 

a. Cấu hình mô hình và huấn luyện

![image](https://github.com/user-attachments/assets/ba4b9a0b-ce47-4df8-b742-f46d88745cbb)
 
Kiến trúc vgg_seq2seq sử dụng mạng CNN VGG để trích xuất đặc trưng và decoder kiểu sequence-to-sequence (Seq2Seq) để sinh chuỗi ký tự đầu ra.

b. Cấu hình dữ liệu huấn luyện và kiểm thử

![image](https://github.com/user-attachments/assets/cf7f7050-0b14-409e-97fc-15fcacfc77aa)
 
- ‘data_root’ : đường dẫn đến dataset
- ‘train_annotation’ : đường dẫn đến file train.txt
- ‘valid_annotation’ : đường dẫn đến file val.txt

Ảnh sẽ được resize về kích thước cố định để đưa vào mô hình: 

![image](https://github.com/user-attachments/assets/076668f5-204d-4625-ac12-ac043fbc34f4)
 
	c. Tham số huấn luyện
 
![image](https://github.com/user-attachments/assets/27c1d5ed-edae-4b6d-b76a-580ad0d41741)

-   iters = 10000: tổng số bước lặp để huấn luyện mô hình.
-   print_every = 50: in loss mỗi 50 bước.
-   valid_every = 500: đánh giá mô hình trên tập val mỗi 500 bước.
-   checkpoint: lưu lại mô hình tạm thời để phục hồi nếu ngắt quãng.
-   export: đường dẫn lưu trọng số cuối cùng sau khi train xong.

d. Các thông số bổ sung

![image](https://github.com/user-attachments/assets/ed228cf2-133d-4d6d-84c0-ab6960344948)
 
-  Sử dụng GPU (cuda) để tăng tốc huấn luyện.
-  Batch size = 16 ảnh/lần.
-  Learning rate ban đầu = 0.0001.

e. Thực hiện huấn luyện
 
![image](https://github.com/user-attachments/assets/445a01f1-dbe4-494e-9bee-314410e63650)

3. Theo dõi và đánh giá
   
Theo dõi train log:
 
![image](https://github.com/user-attachments/assets/303017f4-30c3-4d67-b448-ecea9c0d86f4)
![image](https://github.com/user-attachments/assets/4badc002-0944-4f08-ba86-cae2a767104e)

Đánh giá: 
-   Train loss dao động giảm dần từ ~0.825 xuống ~0.614 – 0.620, cho thấy mô hình tiếp tục học được các đặc trưng cần thiết từ dữ liệu.
-   Valid loss lại dao động quanh mức 0.76 – 0.80, không có xu hướng giảm rõ rệt.
-   Accuracy theo chuỗi đầy đủ (acc full seq) bắt đầu ở mức cao (~78%) nhưng sau đó giảm và dao động quanh 33% – 47%.

-   Accuracy theo ký tự (acc per char) duy trì ổn định từ ~88% – 89%, cao hơn nhiều so với acc full seq, phản ánh mô hình vẫn nhận diện đúng nhiều ký tự, dù đôi khi chưa tái tạo được toàn bộ câu chính xác.

![image](https://github.com/user-attachments/assets/eab544e5-aaa8-4f77-a669-10b9eb99d690)

4. Triển khai mô hình (Inference)
   
Sau khi chúng em hoàn thành huấn luyện mô hình VietOCR, hệ thống nhận diện văn bản được triển khai với giao diện đơn giản sử dụng thư viện Tkinter và Pillow, kết hợp giữa Tesseract OCR (để tách dòng văn bản) và VietOCR (để nhận diện văn bản từng dòng). Quá trình triển khai bao gồm các bước chính như sau:

Bước 1 : Load ảnh

![image](https://github.com/user-attachments/assets/b2767982-b493-4c79-838e-9e4ff8b35b8b)
 
Ảnh của từng dòng được xử lý bằng các bước:
- Chuyển sang grayscale
- Tăng độ tương phản
- Làm nét (sharpen) ảnh
- Nhị phân hóa với ngưỡng cố định (threshold = 160)
- Resize chiều cao ảnh về 128 pixels (chiều rộng tỷ lệ theo)
Mục tiêu: đảm bảo chất lượng ảnh đầu vào cho VietOCR rõ nét, chuẩn kích thước và dễ nhận diện.

Bước 2: Dùng Tesseract để phát hiện từ

 ![image](https://github.com/user-attachments/assets/69b61138-8644-42c8-9094-04c3cf4349ff)

-  Sử dụng thư viện pytesseract.image_to_data() để phân tích ảnh và trích xuất thông tin từng từ: toạ độ (left, top, width, height), văn bản, độ tin cậy (conf).
-  Nhóm các từ theo block_num, par_num, line_num để ghép thành từng dòng văn bản.
  
Bước 3: Load mô hình VietOCR
 
![image](https://github.com/user-attachments/assets/b8e830ee-454b-4410-8a73-e4dc00b71a58)

-  Tải mô hình VietOCR đã huấn luyện (.pth) sử dụng cấu hình vgg_seq2seq.

Bước 4: Chạy OCR từng dòng

![image](https://github.com/user-attachments/assets/7237e052-b9f3-4d23-b52b-cc95d51f51f0)

-   recognized_sentences = [] : Tạo danh sách để lưu kết quả nhận diện của từng dòng.
-  Duyệt từng dòng văn bản đã được gom nhóm (lines là danh sách các DataFrame).
-   Nếu dòng trống (không có từ nào), bỏ qua.
- Tính tọa độ của bounding box bao quanh cả dòng:
+ x1, y1: tọa độ góc trên trái.
+ x2, y2: tọa độ góc dưới phải.
- cropped_line_img : Cắt vùng ảnh chứa dòng văn bản tương ứng từ ảnh gốc.


Bước 5: Hiện thị kết quả
 
![image](https://github.com/user-attachments/assets/336629d2-20e6-4937-b406-4e4314ba1c7e)

Kết quả nhận diện của mô hình nhóm chúng em train: 

![image](https://github.com/user-attachments/assets/5f88ef06-70bf-4900-8ac4-aa8d4c57fc92)
 
5. So sánh và đánh giá với mô hình gốc (pre-trained)

          Để so sánh và đánh giá tổng quát về độ cải thiện giữa mô hình train (fine-tune) so với mô hình gốc (pre-trained) thì chúng em đã lựa chọn đánh giá theo tiêu trí CER (Characters Error Rate) 

a. Chuẩn bị tập dữ liệu đánh giá

- Tập dữ liệu đánh giá cũng sẽ có cấu trúc giống như tập dataset train, nhưng sẽ bao gồm các ảnh kèm nhãn chưa được dùng để train trước đó. 

- Nhóm chúng em đã chuẩn bị 1 tập test gồm 1000 ảnh kèm nhãn tương ứng.

b. Phương pháp đánh giá

-  Sử dụng chỉ số CER (Character Error Rate) – Tỷ lệ lỗi, sai hoặc thiếu ký tự, được tính theo công thức:

CER = Chuỗi dự đoán, Chuỗi đúng(trained) / Số ký tự đúng của (gốc)

-   Mỗi ảnh sẽ được:
+ Nhận diện bởi mô hình gốc và mô hình huấn luyện.
+ Tính CER và ghi lại kết quả.

Kết quả đánh giá:
 
![image](https://github.com/user-attachments/assets/72b77600-ec05-43ab-a492-3172575fa608)

Kết quả cho thấy: 

- CER mô hình gốc : hơn 7% (0,0707)
- CER mô hình train: hơn 6% (0,0659)

Nhận xét: 
-   Mô hình huấn luyện lại thể hiện CER thấp hơn so với mô hình gốc trong nhiều trường hợp.
-   Các từ đặc thù trong tập dữ liệu (ví dụ: tên riêng, dấu tiếng Việt) được nhận diện chính xác hơn sau khi fine-tune.
-   Tuy nhiên, mô hình gốc đôi khi vẫn nhận diện tốt hơn ở một số dòng khó hoặc nhiễu, phản ánh sự ổn định cao của mô hình pretrained.

Kết luận cuối cùng :
        Huấn luyện lại mô hình VietOCR trên tập dữ liệu đặc thù giúp cải thiện độ chính xác tổng thể, đặc biệt là khi tập dữ liệu có nhiều khác biệt so với dữ liệu gốc (phông chữ, chất lượng ảnh, tiếng Việt). Kết quả này chứng minh rằng fine-tuning mô hình trên dữ liệu mục tiêu là cần thiết trong các ứng dụng thực tế.
