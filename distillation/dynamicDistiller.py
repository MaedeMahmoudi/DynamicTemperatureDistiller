import torch
import torch.nn as nn
from distillation.utils import Accuracy, AverageMeter, Hook
from distillation.baseDistiller import BaseDistiller

class DynamicTemperatureDistiller(BaseDistiller):
    def __init__(self, alpha, studentLayer=-2, teacherLayer=-2, t_base=2.0, beta=1.5):
        """
        alpha: وزن بین لاس تقطیر و لاس اصلی (بین 0 تا 1)
        t_base: دمای پایه
        beta: ضریب تأثیر انتروپی
        """
        super(DynamicTemperatureDistiller, self).__init__()
        
        self.alpha = alpha
        self.studentLayer = studentLayer
        self.teacherLayer = teacherLayer
        self.t_base = t_base
        self.beta = beta
        
        self.studentHook = Hook()
        self.teacherHook = Hook()

    def _calculate_dynamic_temperature(self, teacher_logits):
       
        with torch.no_grad():
            # تبدیل لاجیکها به احتمالات
            probs = nn.functional.softmax(teacher_logits, dim=1)
            # محاسبه انتروپی برای هر نمونه
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
            # نرمال‌سازی انتروپی بین 0 و 1
            max_entropy = torch.log(torch.tensor(probs.size(1), dtype=torch.float, device=probs.device))
            entropy_scaled = entropy / max_entropy
            
            # T = T_base + beta * Entropy
            dynamic_T = self.t_base + self.beta * entropy_scaled
            return dynamic_T.unsqueeze(1)  # (Batch, 1)

    def train_step(self, student, teacher, dataloader, optimizer, objective, distillObjective, OneHot=False):
        """
        آموزش شاگرد با تقطیر دانش پویا (دمای متغیر بر اساس انتروپی استاد)
        """
        student.train()
        teacher.eval()
        
        # اتصال هوک‌ها برای گرفتن فعالیت‌های لایه‌های میانی (برای Attention)
        if not self.studentHook.hooked():
            self._setHook(self.studentHook, student, self.studentLayer)
        if not self.teacherHook.hooked():
            self._setHook(self.teacherHook, teacher, self.teacherLayer)

        device = next(student.parameters()).device
        accuracy = Accuracy(OH=OneHot)
        lossMeter = AverageMeter()
        accMeter = AverageMeter()
        
        for _, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

           
            sLogits = student(data)
            tLogits = teacher(data).detach()  
            
            #  گرفتن فعالیت‌های لایه‌های میانی (برای Attention Distillation)
            sAct = self.studentHook.val()
            tAct = self.teacherHook.val()
            
            # تقطیر با دمای پویا روی خروجی نهایی
            #محاسبه دمای پویا بر اساس انتروپی 
            dynamic_T = self._calculate_dynamic_temperature(tLogits) 
            
            # اعمال دما روی لاجیکهای خروجی 
            soft_teacher = nn.functional.softmax(tLogits / dynamic_T, dim=1)
            log_soft_student = nn.functional.log_softmax(sLogits / dynamic_T, dim=1)
            
            # ضریب T^2 برای جبران مقیاس گرادیان‌ها
            mean_T2 = torch.mean(dynamic_T ** 2)
            distill_loss = distillObjective(log_soft_student, soft_teacher) * mean_T2
            
            #  لاس اصلی با برچسب‌های سخت 
            hard_loss = objective(nn.functional.log_softmax(sLogits, dim=1), target)
            
            optimizer.zero_grad()
            batchLoss = (1 - self.alpha) * distill_loss + self.alpha * hard_loss

            batchLoss.backward()
            optimizer.step()
            
            
            lossMeter.update(batchLoss.item(), n=len(data))
            accMeter.update(accuracy(nn.functional.softmax(sLogits, dim=1), target), n=len(data))
        
        return {'Train/Loss': lossMeter.avg,
                'Train/Accuracy': accMeter.avg}