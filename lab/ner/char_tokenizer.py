from transformers import BertTokenizer

class CharTokenizer(BertTokenizer):
    def __init__(self,
            vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=None,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=True,
            strip_accents=None,
            clean_up_tokenization_spaces=True,
            **kwargs
    ):    
        super().__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def _tokenize(self, text, split_special_tokens=False):
        """核心分词方法"""
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(
            text, never_split=self.all_special_tokens if not split_special_tokens else None
        ):
            # If the token is part of the never_split set
            if token in self.basic_tokenizer.never_split:
                split_tokens.append(token)
            else:
                split_tokens.extend(list(token))
        return split_tokens
