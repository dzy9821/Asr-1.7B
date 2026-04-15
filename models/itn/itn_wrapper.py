"""
ITN模型包装器 - FST逆正则化
"""
import os


class ITNProcessor:
    """ITN (Inverse Text Normalization) 处理器"""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "fst_itn_zh")
        self.model_path = model_path
        # TODO: 加载FST模型
        # self.fst = load_fst_model(model_path)

    def process(self, text):
        """
        对文本进行逆正则化处理

        Args:
            text: 输入文本

        Returns:
            normalized_text: 逆正则化后的文本
        """
        # TODO: 使用FST模型处理
        # result = self.fst.process(text)
        # return result

        # 占位符：直接返回输入
        return text
