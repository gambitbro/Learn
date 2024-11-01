# Learn
## lesson
  - You can't change how people feel about you, So don't try. Just live your life and be happy.
  - No matter how hard the past, you can always begin again.
  - suffering is not holding you, you are holding suffering.
  - Win in your mind and you will win in you reality.
  - Don't tell people your plans, show them your results.
  - Change is never painful, only the resistance to change is painful.
  - Nothing is permanent. Don't stress yourself too much, because no matter how the bad situation is... it will change.
  - Live one day at a time. Keep your attention in present time. Have no expectations. Make no judgements. And give up the need to know why things happen as they do. Give it up!
  - Every situation in life is temporary. So, when life is good, make sure you enjoy and receive it fully. And when life is not so good, remember that it will not last forever and better days are on the way.
  - Never be a prisoner of your past. It was just a lesson, not a life sentence.
  - What people think about you is not important. What you think about yourself means everything.
  - Don't compare your life to others. There's no comparison between the sun and the moon. They Shine when it's their time.
  - In the end, people will judge you anyway, so don't live your life impressing others live your life impressing yourself.
  - Don't change so people will like you. Be yourself and the right people will love the real you.
  - All that we are is the result of what we have THOUGHT. The mind is everything. what we think, we BECOME.
  - Life is the most difficult exam. Many people fail because they try to copy others, not realizing that everyone has a different question paper.
  - Don't be jealous with anyone. Don't compete with anyone. Just focus on becoming the best version of yourself.
  - The less you care about what people think, the happier you will be.
  - Begin each day with optimism and end each day with forgiveness. Happiness in life begins and ends within your heart.


## 2024.09.23. 월, DB

  - **⚛️트랜잭션(Transaction)**
    - 트랜잭션은 데이터베이스의 상태를 변화시키는 일련의 작업, 하나의 논리적인 작업 단위로 취급.
    - 일반적으로 트랜잭션은 여러 SQL문으로 구성되며, 그 작업들이 ***모두*** 성공하거나, ***모두*** 실패하는 "**원자성**"을 보장해야 한다.
    - *예시* -> 은행에서 계좌 이체 작업이 있을 때, A계좌에서 100만원을 빼고 B계좌에 100만원을 더하는 작업은 하나의 트랜잭션으로 간주된다. 이 때, A계좌에서만 돈이 빠져나가고 B계좌에 돈이 들어가지 않는 상태는 발생하면 안된다. 전체 작업이 성공하거나, 실패하는 모두 롤백(원래 상태로 돌아감)되어야 한다.

  - **🧰ACID 특성**
    - **A**tomicity(원자성)
      - 트랜잭션이 데이터베이스에 모두 반영되든지, 아니면 전혀 반영되지 않도록 보장하는 특성. 트랜잭션 내의 모든 작업은 하나의 단위로 취급되며, 그 작업들이 전부 성공해야만 데이터베이스에 반영된다.
    
    - **C**onsistency(일관성)
      - 트랜잭션이 실행되기 전과 후에 데이터베이스가 일관된 상태를 유지하도록 보장하는 특성. 트랜잭션이 성공적으로 완료되면, 데이터베이스의 무결성 제약 조건을 모두 만족해야 한다. 예를 들어, 외래 키 제약, 고유 제약 조건 등이 깨지지 않고 유지되어야 한다.

    - **I**solation(격리성)
      - 동시에 여러 트랜잭션이 실행될 때, 각 트랜잭션은 서로 간섭하지 않고 독립적으로 실행되는 것처럼 보여야 하는 특성. 격리성은 여러 트랜잭션이 동일한 데이터를 동시에 읽거나 수정할 때, 서로 간섭하지 않도록 보장한다.

    - **D**urability(지속성)
      - 트랜잭션이 성공적으로 완료되면, 그 결과는 영구적으로 저장되어 시스템이 실패하더라도 보존되는 특성. 예를 들어, 서버가 다운되거나 시스템 오류가 발생해도 트랜잭션에 의해 변경된 데이터는 손실되지 않아야 한다.


## 2024.09.24. 화, DB

### 👻**데이터베이스의 아키텍처(Architecture)**

  - **논리적 아키텍처(Logical Architecture)**
    - 논리적 아키텍처는 데이터베이스 시스템이 사용자의 요청을 처리하는 방식과 관련된 구조. 주로 사용자가 데이터에 접근하는 방법과 데이터를 관리하는 논리적 계층 구조를 나타낸다.
    
    - ***1-티어 아키텍처(Single-tier Architecture)***
      - 모든 데이터베이스 구성 요소가 하나의 시스템 내에 존재하는 아키텍처. 사용자 인터페이스(UI), 애플리케이션, 데이터베이스 서버가 모두 동일한 물리적 장치에 존재하는 경우.

    - ***2-티어 아키텍처(Two-tier Architecture)***
      - 클라이언트-서버 아키텍처로도 알려져 있음. 사용자 인터페이스와 애플리케이션 로직은 클라이언트 측에 있고, 데이터베이스는 별도의 서버에 위치한다. 클라이언트는 데이터베이스와 직접 통신하여 데이터를 요청하고 처리한다.

    - ***3-티어 아키텍처(Three-tier Architecture)***
      - 클라이언트(사용자 인터페이스), 애플리케이션 서버(비즈니스 로직을 처리), 데이터베이스 서버(데이터 저장 및 관리)로 구성된다. 클라이언트는 애플리케이션 서버를 통해 데이터베이스 서버와 통신한다.
      - 보안성, 유지보수성, 확장성이 뛰어나며, 대규모 시스템에서 사용.

    - ***N-티어 아키텍처(N-tier Architecture)***
      - 3티어 아키텍처에서 더 많은 레이어가 추가된 구조. 여기에는 웹 서버, 애플리케이션 서버, 데이터베이스 서버 등 다양한 계층이 포함될 수 있다. 


  - **물리적 아키텍처(physical Architecture)**
    - 물리적 아키텍처는 데이터베이스 시스템이 실제로 데이터가 어떻게 저장되고 관리되는지를 설명하는 구조. 데이터베이스 시스템의 성능과 안정성은 물리적 아키텍처에 크게 좌우된다.

    - ***데이터 저장소(Storage System)***
      - 데이터베이스는 데이터를 **디스크**나 **SSD**같은 저장 장치에 저장한다.
    - ***캐싱 및 버퍼관리***
      - 데이터베이스는 성능을 향상시키기 위해 **메모리**를 사용하여 자주 사용되는 데이터를 캐싱한다. 버퍼 풀(Buffer Pool)이라는 메모리 영역에 데이터를 미리 적재하여, 디스크 I/O를 줄이고 성능을 높인다.
    - ***프로세스 관리***
      - 데이터베이스 시스템은 **멀티스레드 프로세스**를 통해 다중 사용자의 요청을 동시에 처리한다. 각 사용자의 요청은 **독립적인 트랜잭션**으로 처리되며, 트랜잭션 간의 격리성과 동시성을 관리하는 역할을 한다.
    - ***인덱스와 파티셔닝***
      - **인덱스**는 데이터 검색 속도를 높이기 위해 사용. 인덱스는 데이터베이스 테이블의 특정 열에 대해 빠르게 조회할 수 있도록 도와주는 자료구조.
      - **파티셔닝**은 대용량의 데이터를 관리하기 위한 기술. 테이블을 여러 개의 작은 단위로 나누어 관리함으로써 성능과 관리 효율성을 높임.
    - ***복제와 클러스터링***
      - 데이터베이스 시스템은 고가용성과 데이터 무결성을 보장하기 위해 데이터 복제나 클러스터링 기술을 사용한다.
      - **복제(Replication)** : 데이터를 여러 대의 서버에 복사하여, 장애 발생 시 데이터를 보호하고 읽기 작업을 분산시킴.
      - **클러스터링(Clustering)** : 여러 대의 서버가 하나의 데이터베이스를 관리하며, 고가용성과 확장성을 제공.

  - **데이터베이스 시스템의 주요 컴포넌트**
    - 데이터베이스 아키텍처의 핵심 구성 요소들.
    - ***DBMS(Database Management System)***
      - DBMS는 데이터베이스를 생성하고 관리하는 소프트웨어. 데이터의 저장, 검색, 수정, 삭제와 같은 작업을 수행하며, 트랜잭션과 같은 고급 기능을 제공한다. 대표적인 DBMS는 MySQL, PostgreSQL, Oracle, SQL Server등이 있다.
    - ***SQL 엔진***
      - SQL엔진은 사용자가 작성한 SQL 쿼리를 해석하고 실행하는 핵심 모듈이다. 쿼리 최적화, 실행 계획 수립, 트랜잭션 관리 등을 담당.
    - ***스토리지 엔진***
      - 스토리지 엔진은 데이터를 실제로 저장하고 관리하는 모듈. 다양한 유형의 스토리지 엔진이 존재하며, 각기 다른 성능과 특성을 갖고 있다. 예를 들어, MySQL에는 InnoDB와 MyISAM 같은 여러 스토리지 엔진이 있다.
    - ***트랜잭션 관리자***
      - 트랜잭션 관리자는 ACID 특성을 준수하며 트랜잭션을 관리. 트랜잭션 간의 격리성, 원자성을 보장하고, 오류 발생 시 롤백을 통해 데이터 일관성을 유지한다.
    - ***보안 모듈***
      - 보안 모듈은 데이터베이스의 접근 제어와 권한 권리를 담당. 데이터베이스 사용자나 애플리케이션이 허가된 데이터에만 접근할 수 있도록 제어한다.

  - 요약
    - 데이터베이스 아키텍처는 **논리적 아키텍처**와 **물리적 아키텍처**로 나누어 이해할 수 있다. 논리적 아키텍처는 사용자와 데이터베이스 간의 상호작용을 설명하고, 물리적 아키텍처는 데이터가 실제로 어떻게 저장되고 처리되는지를 다룬다. 각종 DBMS 시스템에서는 이러한 아키텍처에 따라 데이터의 일관성, 성능, 확장성, 보안을 보장하며, 최신 클라우드 기반 시스템에서는 자동화와 확장성에 중점을 두고 설계된다.


## 2024.09.25. 수, DB

- SET PERSIST는 MySQL 8.0 이상 버전에서 사용되는 명령어. 시스템 변수를 **영구적으로 변경**하기 위해 사용된다. 즉, 데이터베이스 서버를 재시작하더라도 설정된 시스템 변수가 유지되도록 설정할 수 있다. 이 명령어는 기존의 SET 명령어와 달리, 설정된 값을 **MySQL 설정 파일**에 자동으로 기록해준다.

- 설정된 값은 mysqld-auto.cnf 파일에 기록된다. 이 파일은 MySQL 설정 파일(예: my.cnf)과는 별도로 관리된다.

- SET PERSIST_ONLY는 변수의 값을 **메모리에 반영하지 않고** mysqld-auto.cnf 파일에만 저장하여, **서버가 재시작된 후에만** 변경 사항이 적용되도록 한다.

```sql
// max_connections 값을 영구적으로 변경.
SET PERSIST max_connections = 200;
```


## 2024.09.26. 목, DB

### 😎**사용자 및 권한 관리**

- **사용자 계정 관리**
  - ***사용자 계정 생성*** : DBMS에서는 CREATE USER 명령어를 통해 새로운 사용자 계정을 생성할 수 있다.

    ```sql
    CREATE USER 'username'@'hostname' IDENTIFIED BY 'password';
    ```

    - 'username' : 새로 생성할 사용자 계정의 이름.
    - 'hostname' : 사용자가 어느 호스트에서 접근할 수 있는지 지정. 보통 localhost로 지정하면 로컬 시스템에서만 접근이 가능하다.
    - 'password' : 사용자의 비밀번호를 설정.

  - ***사용자 계정 삭제*** : 불필요한 사용자는 DROP USER 명령어로 삭제 가능.
    ```sql
    DROP USER 'username'@'hostname';
    ```

  - ***사용자 비밀번호 변경*** : ALTER USER
    ```sql
    ALTER USER 'username'@'hostname' IDENTIFIED BY 'new_password';
    ```

- **권한(Privileges) 관리**
  - 각 사용자에게는 데이터베이스 객체에 대해 읽기, 쓰기, 수정, 삭제 등의 권한을 부여할 수 있다. 이 권한은 특정 데이터베이스, 테이블, 열(Column), 또는 다른 객체들에 대해 제한적으로 부여할 수 있다.

  - ***주요 권한 종류***
    - ALL PRIVILEGES : 사용자가 데이터베이스 내에서 모든 작업을 수행할 수 있는 권한을 부여.
    - SELECT : 테이블에서 데이터를 읽을 수 있는 권한.
    - INSERT : 테이블에서 데이터를 삽입할 수 있는 권한.
    - UPDATE : 테이블의 데이터를 수정할 수 있는 권한.
    - DELETE : 테이블에서 데이터를 삭제할 수 있는 권한.
    - CREATE : 새로운 테이블, 데이터베이스, 인덱스 등을 생성할 수 있는 권한.
    - ALTER : 기존 테이블이나 객체의 구조를 변경할 수 있는 권한.
    - DROP : 테이블이나 데이터베이스 등의 객체를 삭제할 수 있는 권한.
    - GRANT OPTION : 다른 사용자에게 자신이 가지고 있는 권한을 부여할 수 있는 권한.

  - ***권한 부여*** : GRANT 명령어 사용.
    ```sql
    GRANT SELECT, INSERT ON database_name.table_name TO 'username'@'hostname';
    ```
    - 'database_name.table_name' : 권한을 부여할 데이터베이스와 테이블.

    - john에게 employees테이블에 대해 읽고 쓰기 권한 부여하기.
      ```sql
      GRANT SELECT, INSERT ON employees TO 'john'@'localhost';
      ```
    - 모든 테이블에 권한 부여
      ```sql
      GRANT SELECT, INSERT ON database_name.* TO 'username'@'hostname';
      ```

  - ***권한 철회(REVOKE)***
    ```sql
    REVOKE SELECT, INSERT ON database_name.table_name FROM 'username'@'hostname';
    ```
    - employees테이블에 대한 john의 INSERT 권한 철회.
      ```sql
      REVOKE INSERT ON employees FROM 'john'@'localhost';
      ```

- **역할(Role) 관리** : 역할(Role)은 여러 권한을 묶어서 사용자에게 부여하는 방식. 이를 통해 동일한 권한 집합을 가진 여러 사용자들을 쉽게 관리할 수 있다. 역할을 사용하면 특정 권한 집합을 가진 사용자를 그룹화하여, 효율적으로 관리할 수 있다.

  - 역할 생성
    ```sql
    CREATE ROLE 'role_name';
    ```
  - 역할에 권한 부여
    ```sql
    GRANT SELECT, INSERT ON database_name.table_name TO 'role_name';
    ```
  - 사용자에게 역할 부여
    ```sql
    GRANT 'role_name' TO 'username'@'hostname';
    ```
  - 역할 삭제
    ```sql
    DROP ROLE 'role_name';
    ```

- **권한 조회 및 확인** : SHOW GRANTS 명령어를 사용하여 특정 사용자가 가지고 있는 권한 목록 확인 가능.
  ```sql
  SHOW GRANTS FOR 'username'@'hostname';
  ```

- **최고 권한 사용자(SUPER User)**
  - 최고 권한 사용자는 모든 사용자에 대한 권한을 수정하고, 데이터베이스 시스템을 제어할 수 있는 특별한 권한을 가진다.
  - 하지만 보안상 root 계정의 사용을 최소화하고, 가능한 범위 내에서 세분화된 권한을 사용자에게 부여하는 것이 권장됨.

  - ***최소 권한 부여*** : 사용자가 작업을 수행하는 데 필요한 최소한의 권한만을 부여한다.
  - ***정기적인 권한 검토*** : 사용자의 권한을 주기적으로 검토하여, 불필요하게 부여된 권한을 철회하고 권한 오남용을 방지한다.
  - ***비밀번호 관리*** : 사용자 계정의 비밀번호는 주기적으로 변경해야 하며, 비밀번호 정책을 설정해 복잡도를 높이고 보안성을 강화한다.
  - ***로그 및 감사*** : 사용자가 수행한 작업을 기록하는 **감사 로그(audit log)** 를 통해 누가 어떤 작업을 했는지 추적할 수 있도록 설정하는 것이 중요하다.


#### **요약예시예시요약**
  - 사용자 생성 및 권한 부여 예시
    - 사용자 alice에게 특정 데이터베이스에서 SELECT와 UPDATE 권한 부여.
      ```sql
      CREATE USER 'alice'@'localhost' IDENTIFIED BY 'securepassword';
      GRANT SELECT, UPDATE ON my_database.* TO 'alice'@'localhost';
      ```
    
    - 사용자 bob에게 모든 권한을 부여한 후 DELETE 권한만 철회.
      ```sql
      CREATE USER 'bob'@'localhost' IDENTIFIED BY 'strongpassword';
      GRANT ALL PRIVILEGES ON my_database.* TO 'bob'@'localhost';
      REVOKE DELETE ON my_database.* FROM 'bob'@'localhost';
      ```

    - 역할 read_only_role을 생성하고, 사용자 charlie에게 이 역할을 부여.
      ```sql
      CREATE ROLE 'read_only_role';
      GRANT SELECT ON my_database.* TO 'read_only_role';
      GRANT 'read_only_role' TO 'charlie'@'localhost';
      ```

  - 요약
    - 사용자 관리 : 데이터베이스에서 각 사용자 계정을 생성하고 비밀번호를 설정.
    - 권한 관리 : GRANT와 REVOKE를 통해 사용자에게 필요한 권한을 부여하거나 철회한다.
    - 역할 관리 : 여러 권한을 묶어서 역할로 관리하고, 사용자에게 역할을 부여함으로써 권한을 효율적으로 관리.
    - 보안 : 최소 권한 원칙을 준수하고, 권한을 정기적으로 검토하여 보안 위험을 줄인다.


- MySQL 아키텍처 흐름
  - 사용자의 요청(클라이언트 요청)
    - MySQL서버는 클라이언트로부터 SQL 쿼리(SELECT, INSERT 등)를 수신한다.
    - 이 과정에서 사용자의 인증 및 권한 확인이 이루어짐.
  - 쿼리 파싱 및 분석
    - SQL명령어가 파서에 의해 구문 분석되어 적절한 명령으로 해석된다.
    - 쿼리의 문법적인 오류가 없는지 확인된다.
  - 쿼리 최적화
    - SQL 최적화기가 쿼리 실행 계획을 수립한다.
    - 최적화 과정에서 인덱스 선택, 조인 방식 등을 결정한다.
  - 스토리지 엔진에 요청
    - 최적화된 쿼리는 **스토리지 엔진**에 전달되어 실제 데이터 작업을 수행한다.
    - MySQL은 다중 스토리지 엔진 구조이므로, 특정 테이블에 대해 어떤 스토리지 엔진을 사용할지 결정한다.
  - 데이터 처리 및 반환
    - 스토리지 엔진에서 데이터를 처리한 후 결과를 SQL 레이어로 반환한다.
    - 최종적으로 결과가 클라이언트로 전달됨.


## 2024.10.21. 월

### 풀업 저항(PULL UP)과 풀다운 저항(PULL DOWN)
  - 풀업 저항(Pull-up Resisotr)
    - 입력 핀이 연결되지 않은 경우 핀을 논리적 '1' 상태로 유지.
    - 논리적 '1' (Vcc).
    - 저항을 Vcc와 입력 핀 사이에 연결.
    
  - 풀다운 저항(Pull-down Resistor)
    - 입력 핀이 연결되지 않은 경우 핀을 논리적 '0' 상태로 유지.
    - 논리적 '0' (GND).
    - 저항을 GND와 입력 핀 사이에 연결.

  - 플로팅 현상 (Floating Phenomenon)
    - 디지털 핀이 연결되지 않거나 불안정한 상태에 있을 때 발생하는 현상. 풀업 또는 풀다운 저항을 사용해 방지할 수 있다.
### 부트로더
  - 기본 개념 이해
    - 부트로더는 시스템이 전원을 켜거나 리셋될 때 가장 먼저 실행되는 코드. 주요 역할은 애플리케이션을 로드하고 실행하기 전에 필요한 초기화 작업을 수행하는 것이다. 커스텀 부트로더를 설계하면 사용자가 부트로더 기능을 확장하거나 수정할 수 있다.

    - 공부가 필요한 내용
      - 부트로더의 역할과 필요성
      - 부트로더와 애플리케이션의 메모리 레이아웃
      - 벡터 테이블(Vector Table) 재배치 방법
      - 부트 모드와 플래시 메모리 구조
  
  - STM32 아키텍처와 부트 모드 이해
    - STM32 MCU는 내부에 기본적으로 제공되는 시스템 부트로더를 갖추고 있으며, 커스텀 부트로더를 올리기 전에 시스템이 어떻게 부팅되는지 이해하는 것이 중요하다.

    - 공부가 필요한 내용
      - STM32의 부트 모드(Flash, RAM, System Memory)
      - 부트 모드를 결정하는 핀(Boot0, Boot1)
      - 플래시 메모리 맵과 섹터 구조
      - 벡터 테이블 재배치(SCB->VTOR 레지스터 설정)

  - STM32 HAL 및 레지스터 수준 프로그래밍 학습
    - STM32의 하드웨어 추상화 레이어(HAL) 라이브러리 또는 레지스터 수준에서 MCU의 동작을 제어하는 법을 학습해야 한다. HAL 라이브러리를 사용하면 부트로더 구현이 상대적으로 간단해지지만, 레지스터 수준 프로그래밍을 익히면 더 많은 유연성을 가질 수 있다.

    - 공부가 필요한 내용
      - HAL API를 사용한 시스템 초기화(클럭, GPIO, UART 등)
      - 플래시 메모리 제어(HAL_FLASH 라이브러리)
      - UART, USB, CAN 등으로 데이터를 송수신하는 방법
      - 벡터 테이블 재배치 및 인터럽트 처리

  - 커스텀 부트로더 설계 및 구현 단계
    - 부트로더는 보통 애플리케이션을 외부에서 받아 플래시에 저장한 후 실행하는 기능을 수행한다.
    - 1) 부트로더 프로젝트 설정
      - STM32CubeIDE에서 새 프로젝트를 생성하고, 부트로더 코드를 작성한다.
        - 부트 모드 설정: MCU의 부트 모드를 설정하는 핀(Boot0)을 설정하여 부트로더가 부팅되도록 한다.
        - 플래시 메모리 섹터 관리: 부트로더가 사용하는 섹터와 애플리케이션이 사용하는 섹터를 구분하여 설정한다.
    - 2) 애플리케이션 로드 기능 구현
      - 부트로더는 UART, USB, I2C, CAN 등의 인터페이스를 통해 새로운 펌웨어를 MCU에 업로드할 수 있다.
        - 데이터 수신: UART 등을 통해 펌웨어 바이너리를 수신하는 코드 작성.
        - 플래시 메모리 쓰기: 수신한 펌웨어를 플래시 메모리에 기록.
          ```c
          HAL_FLASH_Unlock();   // 플래시 메모리 쓰기 가능하도록 언락
          HAL_FLASH_Program(TYPEPROGRAM_WORD, address, data);   // 플래시에 데이터 기록
          HAL_FLASH_Lock();   // 플래시 메모리 다시 잠금
          ```
    - 3) 애플리케이션 실행
      - 벡터 테이블 재배치: 부트로더에서 벡터 테이블을 재배치하여 애플리케이션이 정상적으로 인터럽트를 처리할 수 있도록 한다.
        ```c
        SCB->VTOR = FLASH_APP_START_ADDRESS;    // 애플리케이션의 시작 주소로 벡터 테이블을 재배치
        ```

      - 애플리케이션 진입:
        ```c
        typedef void (*pFunction)(void);
        pFunction jumpToApp;
        uint32_t appJumpAddress = *(__IO uint32_t*) (APP_START_ADDRESS + 4);    // 애플리케이션 시작 주소
        jumpToApp = (pFunction) appJumpAddress;
        jumpToApp();    // 애플리케이션으로 점프
        ```
    - 4) 부트로더 디버깅 및 테스트
      - 부트로더가 예상대로 작동하는지 디버깅이 필요. ST-LINK와 같은 디버거를 사용하여 플래시메모리, 레지스터 상태 등을 확인할 수 있다.

    - 추가로 공부가 필요한 내용
      - 보안: 부트로더의 보안 강화를 위한 암호화 기술 (AES, RSA)공부.
      - OTA 업데이트: 무선 통신(Wi-Fi, BLE)으로 펌웨어를 업데이트하는 방법.
      - USB DFU: STM32의 USB 장치를 이용해 펌웨어를 업데이트하는 방법.

### bss 세그먼트와 data 세그먼트
  - 값이 초기화된 변수는 DS로, 그렇지 않은 변수는 bss로.
    - ```c
      int global variable;  // bss 4 increase
      int global variable = 100; //DS 4 increase
      ```
  - bss, data segment compare test
    - 1. add uninitialized global variable increase 4 in bss
      2. add uninitialized static variable increase 4 in bss
      3. add initialize global variable increase 4 in DS
      4. add initailize static variable increase 4 in DS

    - how to make and see result file
      - you can see the size of BSS segment and DATA segment. 
      - ```bash
        $ gcc src_file.c -o result
        $ size result
        ```
      - size 명령은 생성된 오브젝트 파일에 대한 텍스트, 데이터, BSS 세그먼트의 크기(바이트)를 리턴한다.


## 2024.10.29. 화

### Callback 함수
  - 정의: 콜백 함수는 다른 함수의 인자로 전달되어, 그 함수 내에서 특정 시점에 실행되는 함수.
  - 목적: 특정 작업이 완료된 후에 추가적인 작업을 수행하거나, 이벤트가 발생했을 때 실행되도록 한다.
  - 예시: 버튼 클릭 이벤트, 비동기 네트워크 응답, 파일 읽기/쓰기 완료 등.
  - 활용
    - 1. 비동기 처리: 네트워크 통신, 파일 처리, 타이머 등 시간이 오래 걸리는 작업을 처리할 때, 해당 작업이 완료된 후 수행할 동작을 콜백 함수로 정의.
    - 2. 이벤트 처리: 버튼 클릭, 마우스 이동 등 사용자 인터페이스에서 발생하는 이벤트에 대응하기 위해 콜백 함수가 사용됨.
    - 3. 함수의 동작 커스터마이징: 어떤 함수가 다른 함수에 특정 동작을 위임하고자 할 때, 콜백 함수를 전달하여 동작을 커스터마이징할 수 있다(모듈화하여 넘겨주기 편함, 추후 수정도 편함).

  - Python에서의 콜백 함수
    ```python
    def my_callback(data, callback):
      print("Callback function called with result: ", result)
    
    def process_data(data, callback):
      result = data * 2
      # 작업이 완료되면 콜백 함수 호출
      callback(result)
    
    # 콜백 함수를 인자로 전달
    process_data(10, my_callback)
    ```
  
  - JavaScript에서의 콜백 함수
    ```JS
    function doTask(task, callback)
    {
      console.log("Doing task...");
      // 작업 완료 후 콜백 호출
      callback("Task complete!");
    }

    function onTaskComplete(message)
    {
      console.log("Callback message:", message);
    }

    // 콜백 함수를 인자로 전달
    doTask("My Task", onTaskComplete);
    ```

  - C 언어에서의 콜백 함수
    ```c
    #include <stdio.h>

    // 콜백 함수의 타입 정의
    typedef void (*callback_t)(int);

    // 콜백 함수
    void my_callback(int result)
    {
      printf("Callback called with result: %d\n", result);
    }

    // 콜백 함수를 사용하는 함수
    void perform_operation(int num, callback_t callback)
    {
      int result = num * 2;
      // 작업 완료 후 콜백 호출
      callback(result);
    }

    int main(void)
    {
      // 콜백 함수 포인터를 전달
      perform_operation(5, my_callback);
      
      return 0;
    }
    ```

###
###