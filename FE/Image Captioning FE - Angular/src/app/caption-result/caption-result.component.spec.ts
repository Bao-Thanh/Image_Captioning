import { async, ComponentFixture, TestBed } from '@angular/core/testing';
import { IonicModule } from '@ionic/angular';

import { CaptionResultComponent } from './caption-result.component';

describe('CaptionResultComponent', () => {
  let component: CaptionResultComponent;
  let fixture: ComponentFixture<CaptionResultComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ CaptionResultComponent ],
      imports: [IonicModule.forRoot()]
    }).compileComponents();

    fixture = TestBed.createComponent(CaptionResultComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }));

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
