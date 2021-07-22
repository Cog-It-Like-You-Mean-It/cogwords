def database():
    def __init__():
        self.images = coco_data["images"]
        self.captions = coco_data["annotations"]
    
        # image-ID -> [cap-ID-1, cap-ID-2, ...]
        # image-ID -> url
        # caption-ID -> image-ID
        # caption-ID -> caption (e.g. 24 -> "two dogs on the grass")
        # create dictionary that maps image-ID to caption-ID
        self.iid_to_cid = {}
        self.iid_to_url = {}
        self.cid_to_iid = {}
        self.cid_to_caption = {}
        
        for cap in self.captions:
            self.cid_to_caption[cap['id']] = cap['image_id']
            self.cid_to_iid[cap['id']] = cap['caption']

        for img in self.images:
            self.id_to_url[img['id']] = img['url']
