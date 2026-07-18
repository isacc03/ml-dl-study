# 10-2 트랜스포머로 상품 설명 요약하기
# Source: https://colab.research.google.com/drive/1XxQqisL1Zx2QRr_G02tWsajyCAYNkfHQ?usp=drive_link
# Generated from the original Colab notebook.

# %% Cell 1
# 실행 전 설치: pip install -U transformers==4.57.6

# %% Cell 2
# 깃허브에서 위젯 상태 오류를 피하기 위해 진행 표시줄을 나타내지 않도록 설정합니다.
from transformers.utils import logging

logging.disable_progress_bar()

# %% Cell 3
# 허깅 페이스의 transformers 패키지에서 pipeline 함수 import 하기
from transformers import pipeline

# %% Cell 4
# 기본 모델을 사용해 텍스트 요약 파이프라인을 생성합니다.
pipe = pipeline(task='summarization', device=0)

# %% Cell 5
# 모델을 직접 지정해서 텍스트 요약 파이프라인을 생성합니다.
pipe = pipeline(task='summarization', model='sshleifer/distilbart-cnn-12-6', device=0)

# %% Cell 6
# 텍스트 샘플을 파이프라인에 전달합니다.
sample_text = """Vincent Willem van Gogh was a Dutch Post-Impressionist
painter who is among the most famous and influential figures in the history
of Western art. In just over a decade, he created approximately 2100
artworks, including around 860 oil paintings, most of them in the last two
years of his life. His oeuvre includes landscapes, still lifes, portraits,
and self-portraits, most of which are characterised by bold colours and
dramatic brushwork that contributed to the rise of expressionism in modern
art. Van Gogh's work was beginning to gain critical attention before he died
from a self-inflicted gunshot at age 37. During his lifetime, only one of
Van Gogh's paintings, The Red Vineyard, was sold.
"""
pipe(sample_text)

# %% Cell 7
# 사전 훈련된 KoBART 요약 모델과 토크나이저를 불러와 요약 파이프라인을 생성합니다.
kobart = pipeline(task='summarization', model='EbanLee/kobart-summary-v3', device=0)

# %% Cell 8
# 텍스트 샘플을 파이프라인에 전달합니다.
ko_text = """하나, ‘입문자 맞춤형 7단계 구성’을 따라가며 체계적으로 반복하는 탄탄한 학습 설계!
이 책은 데이터 분석의 핵심 내용을 7단계에 걸쳐 반복 학습하면서 자연스럽게 머릿속에 기억되도록 구성했습니다. [핵심 키워드]와 [시작하기 전에]에서 각 절의 주제에 대한 대표 개념을 워밍업하고, 이론과 실습을 거쳐 마무리에서는 [핵심 포인트]와 [확인 문제]로 한번에 복습합니다. ‘혼자 공부할 수 있는’ 커리큘럼을 그대로 믿고 끝까지 따라가다 보면 데이터 분석 공부가 난생 처음인 입문자도 무리 없이 책을 끝까지 마칠 수 있습니다!
둘, 실제로 일어날 법한 흥미로운 스토리에 담긴 문제를 직접 해결하며 익히는 ‘진짜’ 데이터 분석!
현장감 넘치는 스토리를 통해 데이터를 다루는 방법을 알려 주어 ‘파이썬’과 ‘데이터’가 낯설어도 몰입감 있는 학습을 할 수 있도록 구성했습니다. 이 책에서는 API와 웹 스크래핑을 통해 실제 도서관 데이터와 온라인 서점 웹사이트에서 데이터를 가져오는 등 내 주변에 있는 데이터를 직접 수집할 수 있는 방법을 가이드합니다. 또한 판다스, 넘파이, 맷플롯립 등 데이터 분석에 유용한 각종 파이썬 라이브러리를 활용해 보며 코딩 감각을 익히고, 핵심 통계 지식으로 기본기를 탄탄하게 다질 수 있습니다. 마지막에는 분석을 바탕으로 미래를 예측하는 머신러닝까지 맛볼 수 있어 데이터 분석의 처음부터 끝까지 제대로 배울 수 있습니다.
셋, ‘혼공’의 힘을 실어줄 동영상 강의와 혼공 학습 사이트 지원!
책으로만 학습하기엔 여전히 어려운 입문자를 위해 저자 직강 동영상도 지원합니다. 또한 학습을 하며 궁금한 사항은 언제든지 저자에게 질문할 수 있도록 학습 사이트를 제공합니다. 저자가 질문 하나하나에 직접 답변을 달아 주는 것은 물론, 관련 최신 기술과 정보도 얻을 수 있습니다. 게다가 혼자 공부하고 싶지만 정작 혼자서는 자신 없는 사람들을 위해 혼공 학습단을 운영합니다. 혼공 학습단과 함께하면 마지막까지 포기하지 않고 완주할 수 있을 것입니다.
▶ https://hongong.hanbit.co.kr
▶ https://github.com/rickiepark/hg-da
넷, 언제 어디서든 가볍게 볼 수 있는 혼공 필수 [용어 노트] 제공!
꼭 기억해야 할 핵심 개념과 용어만 따로 정리한 [용어 노트]를 제공합니다. 처음 공부하는 사람들이 프로그래밍을 어려워하는 이유는 낯선 용어 때문입니다. 그러나 어려운 것이 아니라 익숙하지 않아서 헷갈리는 것이므로, 용어나 개념이 잘 생각나지 않을 때는 언제든 부담 없이 [용어 노트]를 펼쳐 보세요. 제시된 용어 외에도 새로운 용어를 추가하면서 자신만의 용어 노트를 완성해가는 과정도 또 다른 재미가 될 것입니다.
"""

kobart(ko_text)

# %% Cell 9
# kobart 파이프라인에 저장된 모델의 설정 정보를 확인합니다.
print(kobart.model.config)

# %% Cell 10
# kobart 모델의 어휘 사전 크기를 확인합니다.
print(kobart.tokenizer.vocab_size)

# %% Cell 11
# 단순하게 토크나이저 객체에 len 함수를 적용하여 어휘 사전의 크기를 확인합니다.
len(kobart.tokenizer)

# %% Cell 12
# vocab 속성을 이용하여 전체 어휘 사전을 가져올 수도 있습니다.
vocab = kobart.tokenizer.vocab
len(vocab)

# %% Cell 13
# 맨 처음 토큰 10개를 확인해봅니다.
list(vocab.items())[:10]

# %% Cell 14
# 샘플 텍스트를 직접 토큰으로 나누어 출력해봅니다.
tokens = kobart.tokenizer.tokenize('혼자 만들면서 공부하는 딥러닝')
print(tokens)

# %% Cell 15
# 각 토큰을 토큰 아이디로 변환합니다.
kobart.tokenizer.convert_tokens_to_ids(tokens)

# %% Cell 16
# 마찬가지로 토큰 아이디로 변환합니다. 맨 앞과 뒤에 0과 1은 문자열의 시작과 끝을 나타냅니다.
tokens_ids = kobart.tokenizer.encode('혼자 만들면서 공부하는 딥러닝')
print(tokens_ids)

# %% Cell 17
# 토큰 아이디를 다시 토큰으로 변환하여 확인합니다.
tokens = kobart.tokenizer.convert_ids_to_tokens(tokens_ids)
print(tokens)

# %% Cell 18
# 토큰 아이디 리스트를 다시 원래 문자열로 복원하여 출력합니다.
kobart.tokenizer.decode(tokens_ids)

# %% Cell 19


