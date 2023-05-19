import { CaptionService } from './../Services/caption.service';
import { Component, ElementRef, ViewChild } from '@angular/core';
import { Camera } from '@ionic-native/camera/ngx';
import { Platform } from '@ionic/angular';
import { ToastController } from '@ionic/angular';

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
  searchGenerated: any;
  searchText: any;

  constructor(
    private camera: Camera,
    private platform: Platform,
    private toastController: ToastController,
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
    this.searchGenerated = false;

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

  searchCaption() {
    this.searchGenerated = true

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

  async generateCaption_view() {
    if(!this.imgUrl) { return; }
    this.captionGenerated = true;
    this.resultGenerated = false;

    let img_path

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

  async generateCaption_read() {
    if (!this.caption) {
      return;
    }
  
    try {
      // Check if the platform is desktop
      if (!this.platform.is('cordova')) {
        if ('speechSynthesis' in window) {
          const utterance = new SpeechSynthesisUtterance(this.caption);
          utterance.lang = 'en-US'; // Set the language to English
          speechSynthesis.speak(utterance);
        } else {
          console.log('Text-to-speech is not supported on this platform.');
        }
      } else {
        console.log('Text-to-speech is not supported on mobile platforms.');
      }
    } catch (error) {
      console.log(error);
    }
  }
  
  async copyCaption() {
    if (!this.caption) {
      return;
    }
  
    try {
      if (this.platform.is('cordova')) {
        // Copy functionality is not supported on mobile platforms
        console.log('Copying to clipboard is not supported on mobile platforms.');
        return;
      }
  
      const textarea = document.createElement('textarea');
      textarea.value = this.caption;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
  
      console.log('Caption copied to clipboard');
  
      const toast = await this.toastController.create({
        message: 'Caption copied to clipboard',
        duration: 2000, // Adjust the duration as needed
        position: 'bottom' // Set the toast position
      });
      toast.present();
    } catch (error) {
      console.log('Error copying caption to clipboard:', error);
      // Handle error or show error message to the user
    }
  }
  
  onSearchChange(event: any) {
    const searchTerm = event.detail.value;
    // Perform search based on the entered search term
    // Add your search logic here
    console.log('Search term:', searchTerm);
  }

  performSearch() {
    // Perform search based on the current value of searchText
    // Add your search logic here
    console.log('Performing search:', this.searchText);
  }
  
  hideCaption() {
    this.captionGenerated = false;
    this.searchGenerated = false;
  }
}
