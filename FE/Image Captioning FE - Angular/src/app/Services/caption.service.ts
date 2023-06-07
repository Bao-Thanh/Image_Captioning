import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { BadInput } from './error/bad-input';
import { NotFoundError } from './error/not-found-error';
import { AppError } from './error/app-error';
import { UnAuthorized } from './error/unauthorized-error';
import { throwError } from 'rxjs';
import { map, catchError } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class CaptionService {

  localUrl: any = 'http://localhost:5000';

  constructor(
      private http: HttpClient
    ) { }
      
  public generateCaption(credentials: string): Observable<any> {
    console.log(credentials);
    
    const url = `${this.localUrl}/predict`;

    return this.http.post(url, { img_path: credentials})
    .pipe(
      map((response: Response) => response),
      catchError(this.handleError)
    );
  }
      
  public search(credentials: string): Observable<any> {
    console.log(credentials);
    
    const url = `${this.localUrl}/search`;

    return this.http.post(url, { search: credentials})
    .pipe(
      map((response: Response) => response),
      catchError(this.handleError)
    );
  }

  private handleError(error: Response) {
    if (error.status === 400) {
      return throwError(new BadInput(error));
    }
    if (error.status === 404) {
      return throwError(new NotFoundError(error));
    }
    if (error.status === 401) {
      return throwError(new UnAuthorized(error));
    }
    return throwError(new AppError(error));
  }
}
