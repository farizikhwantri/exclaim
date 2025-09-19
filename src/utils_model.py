from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn

class CustomModelWithCustomLossFunction(nn.Module):
    def __init__(self, model_name="bert-base-uncased", loss_function=nn.CrossEntropyLoss):
        super(CustomModelWithCustomLossFunction, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 1)
        self.loss_function = loss_function()

    def forward(self, ids, mask, token_type_ids, labels=None):
        outputs = self.model(ids, attention_mask=mask, token_type_ids=token_type_ids)

        output = self.dropout(outputs.pooler_output)
        logits = self.classifier(output)
        
        loss = None
        if labels is not None:
            loss = self.loss_function(logits.view(-1), labels.view(-1).float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )