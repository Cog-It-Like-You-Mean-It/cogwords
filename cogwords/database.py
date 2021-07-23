import numpy as np

class database:
    def __init__(self, file_path):
        from pathlib import Path
        import json
        filename = file_path
        with Path(filename).open() as f:
            coco_data = json.load(f)
        self.coco_data = coco_data
        self.images = coco_data["images"]
        self.captions = coco_data["annotations"]
    
        # image-ID -> [cap-ID-1, cap-ID-2, ...]
        # image-ID -> url
        # caption-ID -> image-ID
        # caption-ID -> caption (e.g. 24 -> "two dogs on the grass")
        # create dictionary that maps image-ID to caption-ID
        from collections import defaultdict
        self.iid_to_cid = defaultdict(list)
        self.iid_to_url = {}
        self.cid_to_iid = {}
        self.cid_to_caption = {}
        self.caption_IDs = []
        self.image_IDs = []
        
        for cap in self.captions:
            self.caption_IDs.append(cap['id'])
            self.cid_to_iid[cap['id']] = cap['image_id']
            self.cid_to_caption[cap['id']] = cap['caption']
            self.iid_to_cid[cap['image_id']].append(cap['id'])

        for img in self.images:
            self.iid_to_url[img['id']] = img['coco_url']

            # append image id to image_IDs
            self.image_IDs.append(img['id'])

        self.caption_IDs = sorted(self.caption_IDs)
        self.image_IDs = sorted(self.image_IDs)
    
    def IDs(self):
        return self.image_IDs, self.caption_IDs

    def get_url(self, img_ID):
        return self.iid_to_url[img_ID]

    def get_caption_ID(self, img_ID):
        return self.iid_to_cid[img_ID]

    def get_caption(self, caption_ID):
        return self.cid_to_caption[caption_ID]

    def get_image_ID(self, caption_ID):
        return self.cid_to_iid[caption_ID]

    def store_word_embeddings(self, word_embeddings):
        self.word_embeddings = word_embeddings # {ID: word_embedding}
        self.reversed_word_embeddings = {key:value for value, key in word_embeddings} # {word_embedding: ID}

    @staticmethod
    def cos_sim(x1, x2):
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    # query by cosine similarities
    def query(self, embedding, k=4):
        # calculate cosine similarities
        cos_sims = []
        for w in self.word_embeddings.values():
            cos_sims.append(self.cos_sim(embedding, w))

        cos_sims = np.concatenate(cos_sims, axis=0)

        # get the top k rows
        top_k = cos_sims.argsort()[0][-k:]
        IDs = []

        for img in top_k:
            IDs.append(self.reversed_word_embeddings[img])
        # # get the top k labels
        # top_k_labels = [labels[i] for i in top_k]
        
        return IDs
    
    def display_image(self, image_ID):
        from image import download_image

        download_image(self.get_url(image_ID)).show()