import os
import time
import copy
import random
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt 


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn import metrics

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from lime import lime_text
from lime.lime_text import LimeTextExplainer
from torchvision.transforms.functional import to_pil_image
import webbrowser


from CLIP.clip import clip
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


from helper_functions import *
from text_normalizer import *

torch.cuda.empty_cache() #free up some memory

class Args:
    bs = 64 #! 
    maxlen = 77
    epochs = 20
    lr = 1e-4
    vmodel = 'vit14'
    net = 'MMNetwork'


class MMNetwork(nn.Module):
    def __init__(self, vdim, tdim, n_cls):
        super(MMNetwork, self).__init__()
        self.vfc = nn.Linear(vdim, 256)
        self.bigru = nn.LSTM(tdim, 256, 1, bidirectional=False, batch_first=True, bias=False)
        self.mfc1 = nn.Linear(512, 256)

        self.cf1 = nn.Linear(256, 1)
        self.cf2 = nn.Linear(256, 1)
        self.cf3 = nn.Linear(256, 1)
        self.cf4 = nn.Linear(256, 1)
        self.cf5 = nn.Linear(256, 1)

        self.act = nn.ReLU()
        self.vdp = nn.Dropout(0.2)
        self.tdp = nn.Dropout(0.2)

    def forward(self, vx, tx, masks=None):
        vx = self.vdp(self.act(self.vfc(vx)))
        _, hidden_tx = self.bigru(tx)
        tx_feat = self.tdp(hidden_tx[0])
        tx_feat = tx_feat.repeat(vx.size(0), 1, 1).mean(dim=1)
        mx = self.act(self.mfc1(torch.cat((vx, tx_feat), dim=1)))

        return (
            torch.sigmoid(self.cf1(mx)),
            torch.sigmoid(self.cf2(mx)),
            torch.sigmoid(self.cf3(mx)),
            torch.sigmoid(self.cf4(mx)),
            torch.sigmoid(self.cf5(mx)),
        )
    

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()


def image_exists(row, image_dir):
    return os.path.exists(os.path.join(image_dir, row['file_name']))


class Explainable_Classifier:
    def __init__(self):
        self.args = Args()
        seed = 42 # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = _Tokenizer()


    def tokenize(self, text, context_length: int = 77):
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self._tokenizer.encode(text)[:context_length-2] + [eot_token]
        result = torch.zeros(context_length, dtype=torch.long)
        mask = torch.zeros(context_length, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)
        mask[:len(tokens)] = 1
        return result, mask

    def evaluate(self, model, loader, clip_model, criterion):
        model.eval()
        test_loss = 0
        preds, labels = [[] for _ in range(5)], [[] for _ in range(5)]

        with torch.no_grad():
            for img_inps, txt_tokens, masks, *lbls in loader:
                img_inps, txt_tokens, masks = img_inps.to(self.device), txt_tokens.to(self.device), masks.to(self.device)
                lbls = [lbl.to(self.device) for lbl in lbls]

                img_feats = clip_model.module.encode_image(img_inps)
                txt_feats = clip_model.module.encode_text(txt_tokens)
                outputs = model(img_feats, txt_feats, masks)

                test_loss += sum(criterion(o, l.unsqueeze(1).float()).item() for o, l in zip(outputs, lbls))

                for i in range(5):
                    preds[i].extend((outputs[i] > 0.5).int().cpu().numpy().flatten())
                    labels[i].extend(lbls[i].cpu().numpy().flatten())

        acc = metrics.accuracy_score(labels[0], preds[0])
        f1s = [metrics.f1_score(labels[i], preds[i], average='macro') for i in range(5)]

        return test_loss / len(loader), acc, *f1s


    def train(self, model, optimizer, scheduler, epochs, tr_loader, vl_loader, clip_model, criterion):
        best_model, best_f1, best_epoch = model, 0, 0
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss, corrects, total = 0.0, 0, 0

            for i, (img_inps, txt_tokens, masks, *labels) in enumerate(tr_loader):
                img_inps, txt_tokens, masks = img_inps.to(self.device), txt_tokens.to(device), masks.to(device)
                labels = [lbl.to(self.device) for lbl in labels]
                optimizer.zero_grad()

                with torch.no_grad():
                    img_feats = clip_model.module.encode_image(img_inps)
                    txt_feats = clip_model.module.encode_text(txt_tokens)

                outputs = model(img_feats, txt_feats, masks)
                loss = sum(criterion(o, l.unsqueeze(1).float()) for o, l in zip(outputs, labels))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                running_loss += loss.item()
                corrects += torch.sum((outputs[0] > 0.5).int() == labels[0].data.view_as((outputs[0] > 0.5).int())).item()
                total += len(labels[0])

            scheduler.step()
            val_metrics = self.evaluate(model, vl_loader, clip_model, criterion)
            if val_metrics[2] > best_f1:
                best_model, best_epoch, best_f1 = copy.deepcopy(model), epoch, val_metrics[2]

        return best_model, best_epoch



    ####functions modified or self made for clip_surgery
    ##first we define this function we will need
    def extract_token_embeddings(self, clip_model, img_inps):
        ''' This function does the same process as our classifier, but stops before pooling '''
        with torch.no_grad():
            visual = clip_model.module.visual
            x = visual.conv1(img_inps)        # [B, C, H', W']

            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, HW]
            x = x.permute(0, 2, 1)            # [B, HW, C]
            x = torch.cat(
                [visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1), x], dim=1
            )  # [B, 1 + HW, C]
            x = x + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x)

            for blk in visual.transformer.resblocks:
                x = blk(x)

            x = visual.ln_post(x)
            return x / x.norm(dim=-1, keepdim=True)  # Normalize per token
    #and we redefine their function (in CLIP_surgery.clip.clip.py) to allow for better prompting
    #and changing embedding dimension
    def encode_text_with_prompt_ensemble(self, clip_model, texts, device, prompt_templates=None):
        # using default prompt templates for ImageNet
        if prompt_templates == None:
            prompt_templates = ['a bad photo of a {}.', 'a photo of the hard to see {}.', 'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

        text_features = []
        for t in texts:
            prompt_templates, hard_prompts = self.give_templates(t) #create soft (format string) and hard (complete sentence) prompts
            prompted_t = [template.format(t) for template in prompt_templates]
            prompted_t.extend(hard_prompts)
            prompted_t = torch.stack([self.tokenize(p)[0] for p in prompted_t]).to(device) #changed this a bit because our tokenizer doesn't accept lists
            class_embeddings = clip_model.encode_text(prompted_t)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)
        text_features = torch.stack(text_features, dim=1).to(device).t()

        return text_features

    def project_txt_to_shared_space(self, clip_model, text_embeddings):
        text_proj = clip_model.module.text_projection
        return text_embeddings @ text_proj
    def project_img_to_shared_space(self, clip_model, image_embeddings):
        image_proj = clip_model.module.visual.proj
        image_embeddings = image_embeddings[:, 1:, :]  # Remove CLS token first
        projected_img = image_embeddings @ image_proj
        return projected_img

    def give_templates(self, text):
        ''' Give prompt templates based on the give text'''

            
        prompt_template = ["an internet meme containing {}", " an image showing {}", "a sarcastic meme containing {}", 
                        'a cartoon showing {}.', "a bad photo depicting {}", "a pixelated image which shows {}"]

        if text=="misogyny":
            hard_prompts = ["a misogynistic image", "a misogynistic internet meme", "a joke that degrades women",
            " an offensive image towards women", "a sarcastic meme containing misogyny", 
            "a social media image that attacks women", "a demeaning post about women",
            "a viral image mocking women", "a disrespectful meme containing misogyny", 
            "a crude joke image containing misogyny", "a meme depicting misogynistic content",
            "a sarcastic meme directed at women"]
        elif text=="shaming":
            hard_prompts = ["a meme that humiliates women","a joke image making fun of women's appearance",
            "a degrading post mocking women's behavior","a social media post that body shames women",
            "an image meant to embarrass women publicly", "a cartoon that ridicules women's intelligence",
            "an internet joke that slut-shames women", "a viral image that shames women for their choices",
            "a sarcastic meme targeting women's appearance","a disrespectful post blaming women for social problems"]
            
        elif text == "violence":
            hard_prompts = ["a meme that depicts violence against women", "an image glorifying abuse of women",
            "a cartoon showing a women being attacked", "a joke post that normalizes violence toward women",
            "a social media image mocking domestic abuse", "an offensive meme trivializing assault against women",
            "a degrading image portraying a woman being hurt", "a joke image suggesting it's funny to harm women",
            "a viral post that justifies mistreating women","a visual showing gender violence"]

        elif text=="stereotype":
            hard_prompts = ["a meme that reinforces stereotypes about women","a social media post promoting sexist views about women",
            "a joke image based on gender stereotypes","a cartoon that reflects traditional views of women",
            "an image suggesting limited roles for women","a viral post implying how women should behave",
            "a sarcastic meme that reduces women to a stereotype","an offensive post expressing biased ideas about women",
            "a visual that promotes outdated beliefs about women", "a joke that relies on generalizations about women"]
        
        elif text=="objectification":
            hard_prompts = [ "a meme that objectifies women", "a joke image reducing women to their appearance",
            "a post portraying women as objects", "a cartoon that treats women as things to be looked at",
            "a visual that dehumanizes women", "an internet meme emphasizing the physical over the personal in women",
            "a social media image that sexualizes women", "an offensive post that commodifies women",
            "a viral image that presents women as sexual objects", "an objectifying meme"]
        else :
            hard_prompts = []
        return prompt_template, hard_prompts

    ########end of modified clip surgery functions


def main():
    status = 'map' #'train' #'test'

    xai_classifier = Explainable_Classifier()
    

    batch_size, init_lr, epochs, vmodel = xai_classifier.args.bs, xai_classifier.args.lr, xai_classifier.args.epochs, xai_classifier.args.vmodel

    clip_nms = {'vit32': 'ViT-B/32', 'vit16': 'ViT-B/16', 'vit14': 'ViT-L/14', 'rn50': 'RN50'}
    clip_dim = {'vit32': 512, 'vit16': 512, 'vit14': 768, 'rn50': 1024}

    clip_model, _ = clip.load(clip_nms[vmodel], jit=False)
    clip_model.float().eval()
    clip_model = nn.DataParallel(clip_model)
    input_resolution = clip_model.module.visual.input_resolution
    dim = clip_dim[vmodel]

    transform_config = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_resolution),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758])
        ]),
        'test': transforms.Compose([
            transforms.Resize((input_resolution, input_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758])
        ])
    }

    data_dir = 'data/'
    train_img_dir = 'data/training_images'
    test_img_dir = 'data/test_images'

    tr_df, vl_df, ts_df = pd.read_csv(data_dir + 'train.tsv', sep='\t'), pd.read_csv(data_dir + 'validation.tsv', sep='\t'), pd.read_csv(data_dir + 'test.tsv', sep='\t')
    tr_df = tr_df[tr_df.apply(lambda row: image_exists(row, train_img_dir), axis=1)].reset_index(drop=True)
    vl_df = vl_df[vl_df.apply(lambda row: image_exists(row, train_img_dir), axis=1)].reset_index(drop=True)

    tr_data = CustomDatasetFixed(tr_df, 'training', transform_config['test'], preprocess, xai_classifier.tokenize, xai_classifier.args.maxlen)
    vl_data = CustomDatasetFixed(vl_df, 'training', transform_config['test'], preprocess, xai_classifier.tokenize, xai_classifier.args.maxlen)
    ts_data = CustomDatasetFixed(ts_df, 'test', transform_config['test'], preprocess, xai_classifier.tokenize, xai_classifier.args.maxlen)

    tr_loader = DataLoader(tr_data, shuffle=True, num_workers=0, batch_size=batch_size)
    vl_loader = DataLoader(vl_data, num_workers=0, batch_size=batch_size)
    ts_loader = DataLoader(ts_data, num_workers=0, batch_size=batch_size)

    model = MMNetwork(dim, dim, 1).to(xai_classifier.device)
    criterion = nn.BCELoss()

    if status == 'train':
        optimizer = optim.Adam(model.parameters(), init_lr, betas=(0.99, 0.98), weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 10, 15], gamma=0.5)
        model_ft, best_epoch = train(model, optimizer, scheduler, epochs, tr_loader, vl_loader, clip_model, criterion)
        torch.save(model_ft.state_dict(), f'model_{xai_classifier.args.net}_{vmodel}.pt')

    elif status == 'test':
        model.load_state_dict(torch.load(f'saved_models/trained_model_{xai_classifier.args.net}_{xai_classifier.args.vmodel}.pt'))
        model.to(xai_classifier.device)
        model.eval()
        ts_loss, ts_acc, *f1_scores = evaluate(model, ts_loader, clip_model, criterion)
        print(f'Test Results → Loss: {ts_loss:.4f} | Accuracy: {ts_acc * 100:.2f}% | F1 Score: {f1_scores[0] * 100:.2f}%')
    
    elif status == 'map': #this does saliency + CLIP surgery
        model.load_state_dict(torch.load(f'saved_models/trained_model_{xai_classifier.args.net}_{xai_classifier.args.vmodel}.pt'))
        model.to(xai_classifier.device)
        model.train() #apparently model needs to be in train mode for saliency maps to work
        # Saliency map computation can be inserted here
        select_one_specific_image = True    
        if select_one_specific_image :
            # Index of the image you want
            idx = 538  # need image number -2 for some unkown reason

            # Get the item directly from the dataset
            img_inp, txt_token, mask, lbl1, lbl2, lbl3, lbl4, lbl5 = ts_data[idx]

            # Convert to batch dimension (1, ...) and move to xai_classifier.device
            img_inps = img_inp.unsqueeze(0).to(xai_classifier.device)
            txt_tokens = txt_token.unsqueeze(0).to(xai_classifier.device)
            masks = mask.unsqueeze(0).to(xai_classifier.device)
            labels1 = torch.tensor([lbl1]).to(xai_classifier.device)
            labels2 = torch.tensor([lbl2]).to(xai_classifier.device)
            labels3 = torch.tensor([lbl3]).to(xai_classifier.device)
            labels4 = torch.tensor([lbl4]).to(xai_classifier.device)
            labels5 = torch.tensor([lbl5]).to(xai_classifier.device)
        else :
            img_inps, txt_tokens, masks, labels1, labels2, labels3, labels4, labels5 = next(iter(ts_loader))

        img_inps, txt_tokens, masks, labels1, labels2, labels3, labels4, labels5 = img_inps.to(xai_classifier.device), \
        txt_tokens.to(xai_classifier.device), masks.to(xai_classifier.device), labels1.to(xai_classifier.device), labels2.to(xai_classifier.device), labels3.to(xai_classifier.device), \
        labels4.to(xai_classifier.device), labels5.to(xai_classifier.device)

        img_inps.requires_grad_()

        #img_inps has shape [batch_size, img_height, img_width, nb_channels]
        print(img_inps.shape)

   
        img_feats = clip_model.module.encode_image(img_inps)
        txt_feats = clip_model.module.encode_text(txt_tokens)

        outputs1, outputs2, outputs3, outputs4, outputs5 = model(img_feats, txt_feats, masks)
        outputs_list = [outputs1, outputs2, outputs3, outputs4, outputs5]
        labels =  ["misogyny", "stereotype",'shaming','objectification', 'violence']
        model.zero_grad()
        
        #recreate originals image from the normalized img_inps (og = normed*std + mean)
        mean = torch.tensor([0.4815, 0.4578, 0.4082]).view(1, 3, 1, 1).to(img_inps.device)
        std = torch.tensor([0.2686, 0.2613, 0.2758]).view(1, 3, 1, 1).to(img_inps.device)
        og_img = img_inps * std + mean
        #for plotting : move to cpu, transform NP array, put in range 0->255 with uint8
        #also need to 
        og_img = og_img.clamp(0, 1).squeeze().permute(1, 2, 0).detach().cpu().numpy() #!
        og_img_uint8 = (og_img * 255).astype('uint8') 

        # Plotting
        img_output_dir = "outputs/"
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Saliency maps \n"
            f"Misogyny score {outputs1[0][0]:.3f} | stereotype {outputs2[0][0]:.3f} | "
            f"shaming {outputs3[0][0]:.3f} | objectification {outputs4[0][0]:.3f} | violence {outputs5[0][0]:.3f}",
            fontsize=25)
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]         
        # Original image in first subplot
        axs[0, 0].imshow(og_img_uint8)
        axs[0, 0].set_title("Original")
        axs[0, 0].axis('off')

        retain_graph = True 
        #create saliency map corresponding to each label
        for n, output in enumerate(outputs_list):
            print(f"iteration {n} of {len(outputs_list)-1}")
            if n==len(outputs_list)-1:
                retain_graph = False #on the last pass we can delete the computation graph to free memory
            output.backward(retain_graph=retain_graph) #backward pass on img_inps starting from output to create all gradients related to this output
            saliency =  img_inps.grad.abs()
            index = 0  # change this if you want to look at another image #! should get rid of this index
            sal = saliency[index].detach().cpu()
            # Convert saliency to grayscale
            saliency_gray = sal.abs().mean(dim=0)  # average over channels
            # Normalize saliency for display
            saliency_gray -= saliency_gray.min()
            saliency_gray /= saliency_gray.max()

            #overlay map & original image

            #plot
            row, col = positions[n + 1]  # shift index since original is at [0,0]
            axs[row, col].imshow(og_img_uint8)  # show RGB image as background
            axs[row, col].imshow(saliency_gray, cmap='hot', alpha=0.7)  # overlay saliency a bit transparent
            axs[row, col].set_title(labels[n])
            axs[row, col].axis('off')   
            img_inps.grad.zero_() #clear gradients for next pass
        
        plt.tight_layout()
        plt.savefig(os.path.join(img_output_dir, f"saliency_map_{idx+2}.png"), bbox_inches='tight')        
        plt.close()
        print(f'saved : saliency_map_{idx+2}.png')

    


        ######starting to implement CLIP surgery
        torch.cuda.empty_cache() #free up some memory
        model.eval() #put back in eval mode after saliency maps are done

        

        from CLIP_Surgery.clip.clip import clip_feature_surgery, get_similarity_map

        ### we will look for the notion "misogyny" in the image
        texts = ["misogyny", "stereotype",'shaming','objectification', 'violence']
    

        # 1. Extract original image features (already used for classification)
        image_features = xai_classifier.extract_token_embeddings(clip_model, img_inps)
        print("extracted img features shape: ", image_features.shape)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)


        # 2. Encode "misogyny" and extract redundant features from empty string 
        text_features = xai_classifier.encode_text_with_prompt_ensemble(clip_model.module, texts, xai_classifier.device)
        redundant_features = xai_classifier.encode_text_with_prompt_ensemble(clip_model.module, [""], xai_classifier.device, prompt_templates=["{}"]) #only give one template for the empty prompt to reduce memory use


        #project everything to shared space
        txt_features = xai_classifier.project_txt_to_shared_space(clip_model, text_features)
        rdd_features = xai_classifier.project_txt_to_shared_space(clip_model, redundant_features)
        img_features = xai_classifier.project_img_to_shared_space(clip_model, image_features)


        print("text_features:", txt_features.shape)           # should be [N, 1024]
        print("redundant_features:", rdd_features.shape) # should be [1, 1024]
        print("image feature : ", img_features.shape)

    

        # 3. Apply feature surgery
        similarity = clip_feature_surgery(img_features, txt_features, rdd_features)

        # 4. Convert to similarity map
        original_shape = img_inps.shape[-2:]  # (H, W)
        similarity_map = get_similarity_map(similarity, original_shape)
        #sim map is the same variable but without grads and on the cpu (for plotting)
        # and unnormalized (pixel values from 0-255 instead of 0-1)
        sim_map =(similarity_map.detach().cpu().numpy() * 255).astype('uint8') 

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("CLIP Surgery \n"
            f"Misogyny score {outputs1[0][0]:.3f} | stereotype {outputs2[0][0]:.3f} | "
            f"shaming {outputs3[0][0]:.3f} | objectification {outputs4[0][0]:.3f} | violence {outputs5[0][0]:.3f}",
            fontsize=25)
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        # Original image in first subplot
        axs[0, 0].imshow(og_img_uint8)
        axs[0, 0].set_title("Original")
        axs[0, 0].axis('off')

        # Draw similarity map
        for b in range(sim_map.shape[0]): #iterate along ?
            for n in range(sim_map.shape[-1]): #iterate along texts
                heatmap = (sim_map[0, :, :, n] * 255).astype('uint8') #shape [H,W] #!
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) #transform to heatmap
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) #convert from BGR to RGB

                row, col = positions[n + 1]  # shift idx since original is at [0,0]
                axs[row, col].imshow(og_img_uint8)
                axs[row, col].imshow(heatmap, alpha=0.6)
                axs[row, col].set_title(texts[n])
                axs[row, col].axis('off')


            print(f'saved : clip_surgery_photo_{idx+2}.png')
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            plt.savefig(os.path.join(img_output_dir, f"clip_surgery_photo_{idx+2}.png"), bbox_inches='tight')
            plt.close()

             # –––––––––––––––––––––––––LIME––––––––––––––––––––––––––––––––––––––––
            #For LIME or any local interpretability method:
            #You choose one specific sample.
            #You generate perturbations of that sample.
            #You observe how predictions change.
            #Then it builds a local surrogate model to explain the prediction.
            # lime librairy : https://lime-ml.readthedocs.io/en/latest/lime.html
          

            # Create an explainer for text
            text_explainer = LimeTextExplainer(class_names=["Non-Misogynistic", "Misogynistic"])

            image_feat = img_feats
        
            # Convert tensor to raw string so that it is readable by human in LIME output 
            tokens = txt_tokens[index]
            #tokens = txt_tokens
            decoded_tokens = [xai_classifier._tokenizer.decoder[t.item()] for t in tokens if t.item() in xai_classifier._tokenizer.decoder]
            raw_text = " ".join(decoded_tokens).replace("<|startoftext|>", "").replace("<|endoftext|>", "")

            def predict_fn(texts): #takes a list of text strings and outputs model predictions in a format suitable for explanation mthods like LIME. It keeps the image fixed (using a precomputed feature) and changes the text inputs IS GENERATING THE PERTUBED TEXT yes it is the text argument that are the pertubaters
                batch_txt = []
                batch_masks = []
                for text in texts:
                    toks, msk = xai_classifier.tokenize(text) # Tokenize each text: get tokens + attention mask
                    batch_txt.append(toks.unsqueeze(0)) # Add batch dimension to tokens and store
                    batch_masks.append(msk.unsqueeze(0)) # Add batch dimension to mask and store
                
                batch_txt = torch.cat(batch_txt).to(xai_classifier.device)# Concatenate all token tensors and masks into a single batch/mask tensor [batch_size, seq_len] and move it to gpu 
                batch_masks = torch.cat(batch_masks).to(xai_classifier.device) # Repeat the same image features for each text in batch
                
                with torch.no_grad(): # Disable gradient for time efficiency 
                    batch_txt_feat = clip_model.module.encode_text(batch_txt) # Encode batch (perturebed data) of texts to features
                    print(image_feat.shape, len(texts))
                    batch_img_feat = image_feat.repeat(len(texts), 1)  # repeat same image features
                    #batch_img_feat = image_feat #only have one image, no need to repeat
                    print(batch_img_feat.shape, batch_txt.shape, batch_masks.shape)
                    output1, _, _, _, _ = model(batch_img_feat, batch_txt_feat, batch_masks) #call the model on all the pertubed text 
                    return torch.cat([1 - output1, output1], dim=1).cpu().numpy() #return and creates a 2-column tensor representing class probabilities for class 0 and class 1 respectively.

            # Generate LIME explanation for the text
            exp = text_explainer.explain_instance(raw_text, predict_fn, num_features=10, labels=[1]) # generating lime explaination on raw_text = OG data, prediction on perturbed text, LIME will highlight 10 most important features

            # Save LIME explanation as HTML
            html_path = f"{img_output_dir}/lime_text_explanation_{idx+2}.html"
            with open(html_path, "w") as f:
                f.write(exp.as_html())

            print(f"LIME explanation saved to {html_path}")

            # Oapen the HTML file in the default web browser automatically
            webbrowser.open(f"file://{html_path}")
            


if __name__ == '__main__':
    main()
