import numpy as np
import pandas as pd

class ReinsuraceTreaty:
    """Classe base para tratados de resseguro"""
    
    def __init__(self, name):
        self.name = name
    
    def apply(self, losses):
        raise NotImplementedError

class QuotaShare(ReinsuraceTreaty):
    """
    Quota Share: ressegurador assume X% de todos os sinistros
    """
    
    def __init__(self, cession_rate, commission_rate=0.25):
        super().__init__(f"Quota Share {cession_rate*100:.0f}%")
        self.cession_rate = cession_rate
        self.commission_rate = commission_rate
    
    def apply(self, losses, premiums=None):
        ceded = losses * self.cession_rate
        retained = losses * (1 - self.cession_rate)
        
        result = {
            'retained_losses': retained,
            'ceded_losses': ceded,
            'total_ceded': ceded.sum(),
            'cession_rate': self.cession_rate
        }
        
        if premiums is not None:
            ceded_premium = premiums * self.cession_rate
            commission = ceded_premium * self.commission_rate
            result['ceded_premium'] = ceded_premium.sum()
            result['commission'] = commission.sum()
            result['net_cost'] = result['ceded_premium'] - result['commission']
        
        return result

class ExcessOfLoss(ReinsuraceTreaty):
    """
    Excess of Loss: ressegurador paga entre retention e limit
    """
    
    def __init__(self, retention, limit):
        super().__init__(f"XL {limit/1e6:.1f}M xs {retention/1e6:.1f}M")
        self.retention = retention
        self.limit = limit
        self.attachment = retention
        self.exhaustion = retention + limit
    
    def apply(self, losses):
        ceded = np.clip(losses - self.retention, 0, self.limit)
        retained = losses - ceded
        
        return {
            'retained_losses': retained,
            'ceded_losses': ceded,
            'total_ceded': ceded.sum(),
            'total_retained': retained.sum(),
            'attachment': self.retention,
            'limit': self.limit,
            'exhaustion': self.exhaustion,
            'n_attaching': (losses > self.retention).sum(),
            'n_exhausting': (losses > self.exhaustion).sum()
        }
    
    def burning_cost(self, losses, years=1):
        result = self.apply(losses)
        return result['total_ceded'] / years
    
    def premium(self, losses, years=1, loading=0.15):
        bc = self.burning_cost(losses, years)
        return bc * (1 + loading)

class StopLoss(ReinsuraceTreaty):
    """
    Stop Loss: protege contra sinistralidade agregada
    """
    
    def __init__(self, attachment_ratio, limit_ratio):
        super().__init__(f"Stop Loss {limit_ratio*100:.0f}% xs {attachment_ratio*100:.0f}%")
        self.attachment_ratio = attachment_ratio
        self.limit_ratio = limit_ratio
    
    def apply(self, losses, premiums):
        total_losses = losses.sum()
        total_premiums = premiums.sum()
        loss_ratio = total_losses / total_premiums
        
        attachment_amount = total_premiums * self.attachment_ratio
        limit_amount = total_premiums * self.limit_ratio
        
        if total_losses <= attachment_amount:
            ceded = 0
        elif total_losses <= attachment_amount + limit_amount:
            ceded = total_losses - attachment_amount
        else:
            ceded = limit_amount
        
        retained = total_losses - ceded
        
        return {
            'total_losses': total_losses,
            'total_premiums': total_premiums,
            'loss_ratio': loss_ratio,
            'ceded_losses': ceded,
            'retained_losses': retained,
            'attachment_ratio': self.attachment_ratio,
            'attachment_amount': attachment_amount
        }

class ReinsuanceProgram:
    """Combina múltiplos tratados"""
    
    def __init__(self, name="Reinsurance Program"):
        self.name = name
        self.treaties = []
    
    def add_treaty(self, treaty):
        self.treaties.append(treaty)
    
    def apply(self, losses, premiums=None):
        original_losses = losses.copy()
        current_losses = losses.copy()
        
        results = {
            'original_losses': original_losses,
            'original_total': original_losses.sum(),
            'treaties': []
        }
        
        for treaty in self.treaties:
            if isinstance(treaty, (ExcessOfLoss, QuotaShare)):
                treaty_result = treaty.apply(current_losses)
                results['treaties'].append({
                    'name': treaty.name,
                    'type': type(treaty).__name__,
                    'result': treaty_result
                })
                current_losses = treaty_result['retained_losses']
        
        results['final_retained'] = current_losses
        results['final_retained_total'] = current_losses.sum()
        results['total_ceded'] = original_losses.sum() - current_losses.sum()
        results['cession_rate'] = results['total_ceded'] / results['original_total']
        
        return results
    
    def summary(self, losses):
        results = self.apply(losses)
        
        print("=" * 70)
        print(f"  {self.name.upper()}")
        print("=" * 70)
        print(f"\nFINANCIAL SUMMARY")
        print(f"  Original losses:     {results['original_total']:>18,.2f}")
        print(f"  Retained losses:     {results['final_retained_total']:>18,.2f}")
        print(f"  Ceded losses:        {results['total_ceded']:>18,.2f}")
        print(f"  Cession rate:        {results['cession_rate']:>18.1%}")
        
        print(f"\nTREATIES IN PROGRAM ({len(self.treaties)}):")
        for i, treaty_data in enumerate(results['treaties'], 1):
            print(f"  {i}. {treaty_data['name']}")
            tr = treaty_data['result']
            if 'total_ceded' in tr:
                print(f"     Ceded: {tr['total_ceded']:,.2f}")
        
        print("=" * 70)
        
        return results