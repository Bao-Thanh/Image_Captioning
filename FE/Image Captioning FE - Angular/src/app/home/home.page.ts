import { CaptionService } from './../Services/caption.service';
import { Component, ElementRef, ViewChild } from '@angular/core';
import { Camera } from '@ionic-native/camera/ngx';
import { Platform } from '@ionic/angular';
import {
  Plugins,
  Capacitor,
  CameraSource,
  CameraResultType
} from '@capacitor/core';

function dataURLtoFile(dataurl, filename) {
  let arr = dataurl.split(','),
      mime = arr[0].match(/:(.*?);/)[1],
      bstr = atob(arr[1]),
      n = bstr.length,
      u8arr = new Uint8Array(n);
  while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
  }
  return new File([u8arr], filename, {type: mime});
}

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage {


  @ViewChild('filePicker') filePickerRef: ElementRef<HTMLInputElement>;
  imgUrl: string;
  caption: string;
  captionGenerated: any;
  resultGenerated: any;
 

  constructor(
    private camera: Camera,
    private platform: Platform,
    private captionService: CaptionService
  ) {}

  ngOnInit() {}

  capTureImageOnDesktop() {
    console.log('Hello');
    Plugins.Camera.getPhoto({
      quality: 90,
      source: CameraSource.Prompt,
      correctOrientation: true,
      height: 320,
      width: 200,
      resultType: CameraResultType.DataUrl
    })
    .then((res) => {
      console.log(res);
      this.imgUrl = res.dataUrl;
    })
    .catch(error => {
      console.log(error);
    });
  }

  captureImageFromCamera() {
    if (Capacitor.isPluginAvailable('Camera')) {
      this.capTureImageOnDesktop();
      return;
    }

    if (this.platform.is("cordova")) {
      this.camera.getPicture({
        sourceType: this.camera.PictureSourceType.CAMERA,
        destinationType: this.camera.DestinationType.DATA_URL
      }).then((res) => {
        this.imgUrl = 'data:image/jpeg;base64,' + res;
        this.captionGenerated = false;
      }).catch((error) => {
        console.log(error);
      });
    } else {
      console.log(this.filePickerRef);
      this.filePickerRef.nativeElement.click();
    }
  }

  captureImageFromGallery() {
    if (this.platform.is("cordova")) {
      this.camera.getPicture({
        sourceType: this.camera.PictureSourceType.PHOTOLIBRARY,
        destinationType: this.camera.DestinationType.DATA_URL
      }).then((res) => {
        this.imgUrl = 'data:image/jpeg;base64,' + res;
        this.captionGenerated = false;
      }).catch((error) => {
        console.log(error);
      });
    } else {
      console.log(this.filePickerRef);
      this.filePickerRef.nativeElement.click();
    }
  }

  onFileChosen(event: Event) {
    const pickedFile = (event.target as HTMLInputElement).files[0];
    if (!pickedFile) {
      return;
    }
    const fr = new FileReader();
    fr.onload = () => {
      const dataUrl = fr.result.toString();
      this.imgUrl = dataUrl;
      // console.log(this.imgUrl);
      this.captionGenerated = false;
    };
    fr.readAsDataURL(pickedFile);
  }

  async generateCaption() {
    if(!this.imgUrl) { return; }
    this.captionGenerated = true;

    let img_path
    // if (typeof this.imgUrl === 'string') {
    //   try {
    //     imageFile = dataURLtoFile(this.imgUrl, 'caption-img.jpeg');        
    //   } catch (error) {
    //     console.log(error);
    //     return;
    //   }
    // } else {
    //   imageFile = this.imgUrl;
    // }

    img_path = this.imgUrl;

    console.log(img_path);

    this.captionService.generateCaption(
      img_path
    )
      .subscribe(
        (response) => {
          console.log(response);
          this.resultGenerated = true
          this.caption = response['caption']
        }, (error) => {
          console.log(error);
        }
      )
  }

  // async generateOTP() {
  //   if(!this.imgUrl) { return; }
  //   this.captionGenerated = true;

  //   let img_path

  //   // if (typeof this.imgUrl === 'string') {
  //   //   try {
  //   //     imageFile = dataURLtoFile(this.imgUrl, 'caption-img.jpeg');        
  //   //   } catch (error) {
  //   //     console.log(error);
  //   //     return;
  //   //   }
  //   // } else {
  //   //   imageFile = this.imgUrl;
  //   // }

  //   img_path = this.imgUrl;

  //   this.captionService.generateOTP(
  //     img_path
  //   )
  //     .subscribe(
  //       (response) => {
  //         console.log(response);
  //       }, (error) => {
  //         console.log(error);
  //       }
  //     )
  // }

  hideCaption() {
    this.captionGenerated = false;
  }
}
