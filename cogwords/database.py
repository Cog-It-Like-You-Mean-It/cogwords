class database:
    def __init__(self):
        from pathlib import Path
        import json
        filename = "data/captions_train2014.json"
        with Path(filename).open() as f:
            coco_data = json.load(f)

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
