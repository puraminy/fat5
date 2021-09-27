
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model1 = T5ForConditionalGeneration.from_pretrained('fat5-base')
tokenizer1 = T5Tokenizer.from_pretrained('fat5-base')

# + [markdown] id="_GVnO2C0ruQx"
# The model has not been fine-tuned on any sensible task except filling the gaps. And even this task is performed strangely - the models continues generating when it should have stopped. 
#
# But we hope that after fine-tuning it will be better. But this is the topic of the next story)

# + colab={"base_uri": "https://localhost:8080/"} id="08zibfjgtNhF" outputId="e53334a4-3a5e-4659-a34e-e06812d3e979"
inputs = tokenizer1('The <extra_id_0> walks in <extra_id_1> park.', return_tensors='pt')
with torch.no_grad():
    hypotheses = model1.generate(
        **inputs, 
        do_sample=True, top_p=0.95, 
        num_return_sequences=3, 
        repetition_penalty=2.5,
        max_length=32,
    )
for h in hypotheses:
    print(tokenizer1.decode(h))

# + colab={"base_uri": "https://localhost:8080/"} id="tsR9lH3_uqF3" outputId="c1659ff2-8f43-4e6e-b21a-9101831a29a3"
inputs = tokenizer1('خرگوش <extra_id_0> جنگل <extra_id_1> کرد.', return_tensors='pt')
with torch.no_grad():
    hypotheses = model1.generate(
        **inputs, 
        do_sample=True, top_p=0.95, 
        num_return_sequences=3, 
        repetition_penalty=2.5,
        max_length=32,
    )
for h in hypotheses:
    print(tokenizer1.decode(h))

# + [markdown] id="2nZt98FYwcex"
# I will save the model on my Google drive to retrieve it later for fine-tuning. 

# + colab={"base_uri": "https://localhost:8080/"} id="iMG9dNShwg9U" outputId="a8008517-ed3f-421e-b6dc-7c15877f55d1"

# + colab={"base_uri": "https://localhost:8080/"} id="_j56QoXBwjCS" outputId="2aa1f55c-2238-460a-8175-92d78069c81e"
#model1.save_pretrained('/drive2/pretrained/mt5/hf/fat5-base-raw')
#tokenizer1.save_pretrained('/drive2/pretrained/mt5/hf/fat5-base-raw')

# + id="57AlTaqpw2Ew"

