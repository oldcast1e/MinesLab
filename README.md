# MinesLab Repository Guide

### 1. 초기 설정

- MinesLab 폴더로 이동 : cd /home/mines/Documents/oldcast1e/MinesLab
- Git 저장소 초기화 : git init
- 브랜치 이름을 main으로 설정 : git branch -M main
- 원격 저장소 연결 : git remote add origin https://github.com/oldcast1e/MinesLab.git

### 2. 사용자 정보 설정 (최초 1회)
git config --global user.name "oldcast1e"
git config --global user.email "kylhs0207@naver.com"

### 3. 커밋 및 업로드 
- 모든 변경사항 스테이징 : git add .
- 커밋 생성 : git commit -m "First commit - MinesLab projects"
- GitHub에 업로드 (main 브랜치) : git push -u origin main
    - Push 시 GitHub 계정 비밀번호 대신 Personal Access Token (PAT) 을 사용
    - 토큰 생성 링크 에서 repo 권한을 가진 토큰을 생성 후, 비밀번호 자리에 입력
